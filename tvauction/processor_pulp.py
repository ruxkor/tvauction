# -*- coding: utf-8; -*-
import sys

from collections import namedtuple,defaultdict
from datetime import datetime
import pulp as pu
import math
import random
from pprint import pprint as pp

pu.LpProblem(name='testing',sense=pu.LpMaximize)

Slot = namedtuple('Slot', ('price','length'))
BidderInfo = namedtuple('BidderInfo', ('budget','length','attrib_min','attribs'))

time_start = datetime.now()

def gwd(slots,bidderInfos):
    '''the winner determination, implemented as a multiple knapsack problem'''
    slots_len = len(slots)
    bidderInfos_len = len(bidderInfos)
    x = defaultdict(dict)
    for i in range(slots_len):
        for j in range(bidderInfos_len):
            x[i][j] = pu.LpVariable('x_%d_%d' % (i,j),cat=pu.LpBinary) 
    
    y = {}
    for j in range(bidderInfos_len):
        y[j] = pu.LpVariable('y_%d' % (j,), cat=pu.LpBinary)
        
    cons = []
        
#    the sum of all assigned ad length has to be at most the length of the slot
    for (i,slot) in enumerate(slots):
        f = sum(bidderInfo.length*x[i][j] for (j,bidderInfo) in enumerate(bidderInfos))
        cons.append(f <= slot.length)
        
#    match the bidders' demands regarding their atttributes
    for (j,bidderInfo) in enumerate(bidderInfos):
        M = slots_len+2
        f = sum(x[i][j] for i in range(slots_len))
        f2 = bidderInfo.attrib_min - f
        cons.append(f <= M*y[j])
        cons.append(f2 <= M*(1-y[j]))
    
#    user can at most spend the maximum price
    for (j,bidderInfo) in enumerate(bidderInfos):
        f = sum(bidderInfo.length*slot.price*x[i][j] for (i,slot) in enumerate(slots))
        cons.append(f <= bidderInfo.budget)
    
#    oovar domain=bool takes already care of min and max bounds

    prob = pu.LpProblem('testing', pu.LpMaximize)
    prob += sum(bidderInfo.budget*y[j] for (j,bidderInfo) in enumerate(bidderInfos))
    
    for con in cons:
        prob += con
    
#    print 'building', datetime.now()-time_start
    return prob

slot_amount = 168/4
bidder_amount = 50/2
bidder_flatten = bidder_amount+1
slots = [Slot(0,120) for i in range(slot_amount)]

rand_increments = [175, 707, 312, 930, 276, 443, 468, 900, 855, 15, 658, 135, 238, 506, 244, 333, 912, 515, 458, 140, 925, 544, 720, 127, 545, 497, 962, 618, 900, 491, 515, 694, 738, 809, 75, 538, 422, 112, 106, 739, 1, 168, 554, 186, 762, 310, 888, 921, 164, 472, 538, 340, 267, 517, 412, 84, 941, 979, 713, 375, 501, 245, 149, 764, 74, 242, 385, 61, 910, 976, 775, 932, 661, 512, 757, 814, 443, 683, 795, 306, 955, 381, 202, 40, 908, 465, 755, 772, 125, 704, 934, 284, 712, 950, 645, 783, 525, 190, 587, 173]
rand_lengths = [60, 15, 45, 105, 90, 105, 15, 60, 30, 75, 90, 105, 105, 105, 60, 15, 75, 30, 60, 60, 30, 60, 30, 30, 90, 60, 90, 45, 45, 120, 105, 60, 60, 120, 90, 15, 45, 45, 45, 60,
 75, 15, 30, 60, 60, 90, 120, 120, 60, 105]
rand_times = [4, 4, 10, 7, 7, 3, 5, 3, 9, 5, 9, 2, 7, 3, 5, 3, 10, 1, 2, 3, 7, 4, 3, 7, 8, 8, 5, 10, 5, 6, 10, 2, 6, 8, 4, 10, 4, 2, 8, 2, 9, 7, 1, 4, 5, 8, 1, 10, 3, 5]

bidderInfos = [
    BidderInfo( (1000+incr)*length*times/120,length,times,[]) 
    for (incr,length,times) 
    in zip(rand_increments,rand_lengths,rand_times)
][:bidder_amount]

# slots = [Slot(1.0,120) for i in range(3)]
# bidderInfos = [
#     BidderInfo(1000,100,1,[]),
#     BidderInfo(1000,100,1,[]),
#     BidderInfo(1000,100,1,[]),
#     BidderInfo(1800,100,3,[]),
# ]

#if bidder_amount>=bidder_flatten:
#    bidderInfos[bidder_flatten:] = [bidderInfos[0]]*len(bidderInfos[bidder_flatten:])

prob = gwd(slots, bidderInfos)
#print 'solving', datetime.now()-time_start

print 'calculating wdp'
solver = pu.GUROBI(msg=False)
prob.solve(solver)
winners = set(int(v.name[2:]) for v in prob.variables() if v.varValue==1 and v.name[:1]=='y')

revenue_raw = pu.value(prob.objective)
prices_raw = dict((w,bidderInfos[w].budget) for w in winners)

print 'raw:\trevenue %d\tprices: %s\n\n' % (revenue_raw,sorted(prices_raw.iteritems()))

print 'calculating vcg',
prices_vcg = {}
for w in winners:
    bidderInfosVcg = bidderInfos[:]
    winner = bidderInfosVcg[w]
    del bidderInfosVcg[w]
    prob_without_w = gwd(slots,bidderInfosVcg)
    prob_without_w.solve(solver)
    revenue_without_w = pu.value(prob_without_w.objective)
    prices_vcg[w] = winner.budget - (revenue_raw-revenue_without_w)
    sys.stdout.write('.')
revenue_vcg = sum(prices_vcg.itervalues())

print '\nvcg:\trevenue %d\tprices: %s\n\n' % (revenue_vcg,sorted(prices_vcg.iteritems()))

iteration = 0
prices_iterations = []
prices_t = prices_vcg.copy()

# build ebpo
prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)

# variables: π_j, m
# constants: ε
pi = dict((w,pu.LpVariable('pi_%d' % (w,), cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=prices_raw[w])) for w in winners)
em = pu.LpVariable('m',cat=pu.LpContinuous)
epsilon = 100

# ebpo objetive function
prob_ebpo += sum(pi.itervalues()) + epsilon*em

# ebpo constraint: π_j - εm <= π_j^vcg 
for w in winners:
    prob_ebpo += pi[w]-em <= prices_vcg[w]

ebpo_solver = pu.GUROBI(msg=False, mip=False)
# abort after 1000 iterations (this should never happen)
for cnt in xrange(1000):
    
    # make a copy of prob_gwd, since we are using it as basis
    prob_sep = prob.deepcopy()
    prob_sep.name = 'sep'
    
    bidderInfos_len = len(bidderInfos)
    
    # build sep t variable
    t = dict((w,pu.LpVariable('t_%d' % (w,), cat=pu.LpBinary)) for w in winners)
    
    # add the 'maximum coalition contribution' to the objective fn
    prob_sep.objective += -sum((prices_raw[w]-prices_t[w])*t[w] for w in winners)
    
    # add all sep constraints, setting y_i <= t_i
    for w in winners:
        prob_sep += prob_sep.variablesDict()['y_%d' % w] <= t[w]
    
    # solve it
    prob_sep.solve(solver)
    
    # save the value: z( π^t )
    revenue_sep = pu.value(prob_sep.objective)
    
    
    blocking_coalition_exists = revenue_sep > sum(prices_t.itervalues())
    if not blocking_coalition_exists: 
        break
    
    # extend and solve the ebpo problem:
    
    blocking_coalition = set(int(b.name[2:]) for b in prob_sep.variables() if b.varValue==1 and b.name[:1]=='y')
    print 'sep:\tblocking: %s' % (sorted(blocking_coalition),)
    
    winners_nonblocking = winners-blocking_coalition
    winners_blocking = winners&blocking_coalition
    
    # add (iteratively) new constraints to the ebpo problem
    prob_ebpo += sum(pi[wnb] for wnb in winners_nonblocking) >= revenue_sep - sum(prices_t[wb] for wb in winners_blocking)
    
    # solve the ebpo (this problem can be formulated as a continous LP).
    prob_ebpo.solve(ebpo_solver)
    
    # updated the π_t list
    prices_t = dict((int(b.name[3:]),b.varValue) for b in prob_ebpo.variables() if b.name[:2]=='pi')
    
print 'core:\trevenue %d\tprices: %s' % (
    pu.value(prob_ebpo.objective),
    ', '.join('%d->%d' % pt for pt in prices_t.iteritems())
)
