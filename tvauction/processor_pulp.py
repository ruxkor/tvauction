# -*- coding: utf-8; -*-
import sys
import logging

from collections import namedtuple, defaultdict
import pulp as pu
import math
import random

Slot = namedtuple('Slot', ('price','length'))
BidderInfo = namedtuple('BidderInfo', ('budget','length','attrib_min','attrib_values'))

class Gwd(object):
    def __init__(self,msg=False,**kw):
        self.solver = pu.GUROBI(msg=msg,**kw)
        self.prob = None
    def generate(self, slots, bidderInfos):
        '''the winner determination, implemented as a multiple knapsack problem'''
        slots_len = len(slots)
        bidderInfos_len = len(bidderInfos)
        
        # x determines whether a bidder can air in a certain slot
        x = pu.LpVariable.dicts('x', (range(slots_len),range(bidderInfos_len)), cat=pu.LpBinary)
        
        # y determines whether a winner has won
        y = pu.LpVariable.dicts('y', (range(bidderInfos_len),), cat=pu.LpBinary)
        
        # initialize constraints
        cons = []
            
        # the sum of all assigned ad lengths has to be at most the length of the slot
        for (i,slot) in enumerate(slots):
            f = sum(bidderInfo.length*x[i][j] for (j,bidderInfo) in enumerate(bidderInfos))
            cons.append(f <= slot.length)
            
        # match the bidders' demands regarding their attributes
        # attrib_values has to be a list with the same length
        # as the slots list
        for (j,bidderInfo) in enumerate(bidderInfos):
            assert slots_len == len(bidderInfo.attrib_values)
            M = sum(bidderInfo.attrib_values)+1
            f = sum(attrib_value*x[i][j] for (i,attrib_value) in zip(range(slots_len),bidderInfo.attrib_values))
            f2 = bidderInfo.attrib_min - f
            cons.append(f <= M*y[j])
            cons.append(f2 <= M*(1-y[j]))
        
        # user can at most spend the maximum price
        for (j,bidderInfo) in enumerate(bidderInfos):
            f = sum(bidderInfo.length*slot.price*x[i][j] for (i,slot) in enumerate(slots))
            cons.append(f <= bidderInfo.budget)
        
        # oovar domain=bool takes already care of min and max bounds
        prob = pu.LpProblem('gwd', pu.LpMaximize)
        prob += sum(bidderInfo.budget*y[j] for (j,bidderInfo) in enumerate(bidderInfos))
        
        for con in cons:
            prob += con
        
        return prob
    
    def solve(self, prob, bidderInfos):
        logging.info('wdp:\tcalculating...')
        
        prob.solve(self.solver)
        winners = set(int(v.name[2:]) for v in prob.variables() if v.varValue==1 and v.name[:1]=='y')
        
        revenue_raw = pu.value(prob.objective)
        prices_raw = dict((w,bidderInfos[w].budget) for w in winners)
        
        logging.info('raw:\trevenue %d\tprices: %s' % (revenue_raw,sorted(prices_raw.iteritems())))
        return (revenue_raw, prices_raw, winners)
        
class Vcg(object):
    def __init__(self, gwd):
        self.gwd = gwd
        pass
    def calculate(self, slots, bidderInfos, revenue_raw, winners):
        logging.info('vcg:\tcalculating...')
        prices_vcg = {}
        for (step,w) in enumerate(winners):
            bidderInfosVcg = bidderInfos[:]
            winner = bidderInfosVcg[w]
            del bidderInfosVcg[w]
            revenue_without_w = self._calculate_step(slots, bidderInfosVcg)
            prices_vcg[w] = winner.budget - (revenue_raw-revenue_without_w)
            logging.debug('calculating vcg - step %d of %d' % (step+1,len(winners)))
        revenue_vcg = sum(prices_vcg.itervalues())
        logging.info('vcg:\trevenue %d\tprices: %s' % (revenue_vcg,sorted(prices_vcg.iteritems())))
        return (revenue_vcg,prices_vcg)
    
    def _calculate_step(self, slots, bidderInfosVcg):
        prob_without_w = self.gwd.generate(slots, bidderInfosVcg)
        prob_without_w.solve(self.gwd.solver)
        revenue_without_w = pu.value(prob_without_w.objective)
        return revenue_without_w
        
class CorePricing(object):
    def __init__(self, gwd):
        self.gwd = gwd 
        self.epsilon = 10000
    
    def solve(self, prob_gwd, bidderInfos, winners, prices_raw, prices_vcg):
        # build ebpo
        prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)
        prices_t = prices_vcg.copy()     
           
        # variables: π_j, m
        # constants: ε
        pi = dict((w,pu.LpVariable('pi_%d' % (w,), cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=prices_raw[w])) for w in winners)
        em = pu.LpVariable('m',cat=pu.LpContinuous)
        epsilon = self.epsilon
        
        # ebpo objetive function
        prob_ebpo += sum(pi.itervalues()) + epsilon*em
        
        # ebpo constraint: π_j - εm <= π_j^vcg 
        for w in winners:
            prob_ebpo += pi[w]-em <= prices_vcg[w]
        
        ebpo_solver = pu.GUROBI(msg=False, mip=False)
        
        # abort after 1000 iterations (this should never happen)
        for cnt in xrange(1000):
            
            # make a copy of prob_gwd, since we are using it as basis
            prob_sep = prob_gwd.deepcopy()
            prob_sep.name = 'sep'
            
            # caching if the len of the bidderInfos list
            bidderInfos_len = len(bidderInfos)
            
            # build sep t variable
            t = dict((w,pu.LpVariable('t_%d' % (w,), cat=pu.LpBinary)) for w in winners)
            
            # add the 'maximum coalition contribution' to the objective fn
            prob_sep.objective += -sum((prices_raw[w]-prices_t[w])*t[w] for w in winners)
            
            # add all sep constraints, setting y_i <= t_i
            for w in winners:
                prob_sep += prob_sep.variablesDict()['y_%d' % w] <= t[w]
            
            # solve it
            prob_sep.solve(self.gwd.solver)
            
            # save the value: z( π^t )
            revenue_sep = pu.value(prob_sep.objective)
            
            blocking_coalition_exists = revenue_sep > sum(prices_t.itervalues())
            if not blocking_coalition_exists: 
                break
            
            # extend and solve the ebpo problem:
            
            blocking_coalition = set(int(b.name[2:]) for b in prob_sep.variables() if b.varValue==1 and b.name[:1]=='y')
            logging.info('sep:\tblocking: %s' % (sorted(blocking_coalition),))
            
            winners_nonblocking = winners-blocking_coalition
            winners_blocking = winners&blocking_coalition
            
            # add (iteratively) new constraints to the ebpo problem
            prob_ebpo += sum(pi[wnb] for wnb in winners_nonblocking) >= revenue_sep - sum(prices_t[wb] for wb in winners_blocking)
            
            # solve the ebpo (this problem can be formulated as a continous LP).
            prob_ebpo.solve(ebpo_solver)
            
            # updated the π_t list
            prices_t = dict((int(b.name[3:]),b.varValue) for b in prob_ebpo.variables() if b.name[:2]=='pi')
            
        revenue_core = sum(prices_t.itervalues())
        logging.info('core:\trevenue %d\tprices: %s' % (
            revenue_core,
            ', '.join('%d->%d' % pt for pt in prices_t.iteritems())
        ))
        return (revenue_core, prices_t)
             

def solve(slots, bidderInfos):
    gwd = Gwd(False,epgap=0.05)
    prob_gwd = gwd.generate(slots, bidderInfos)
    revenue_raw, prices_raw, winners = gwd.solve(prob_gwd, bidderInfos)
    
    vcg = Vcg(gwd)
    revenue_vcg, prices_vcg = vcg.calculate(slots, bidderInfos, revenue_raw, winners)
    
    core = CorePricing(gwd)
    revenue_core, prices_core = core.solve(prob_gwd, bidderInfos, winners, prices_raw, prices_vcg)
    return {
        'winners': list(winners),
        'prices_raw': prices_raw,
        'prices_vcg': prices_vcg,
        'prices_core': prices_core
    }
     
if __name__ == '__main__':
    import json
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-l','--log', dest='loglevel', help='the log level', default='INFO')
    parser.add_option('-s','--scenario', dest='scenario', help='the scenario')
    options = parser.parse_args()[0]
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    if not options.scenario:
        raise ValueError('Scenario needed')
    
    logging.basicConfig(level=numeric_level)
    
    slots, bidderInfos = json.parse(options.scenario)
    res = solve(slots, bidderInfos)
    
    print json.dumps(res,indent=2)    
