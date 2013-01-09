#!/usr/bin/env python
# -*- coding: utf-8; -*-

import logging
import pulp as pu
from collections import OrderedDict, namedtuple

Campaign = namedtuple('Campaign', ('id','cpvs','budget','duration','reach'))
Slot = namedtuple('Slot', ('id', 'price', 'length'))

solver = pu.GUROBI(msg=False)

def calc(slots, campaigns):
    x = pu.LpVariable.dicts('x', (slots.keys(),campaigns.keys()), cat=pu.LpBinary)
    
    cons = []
    for (i, slot) in slots.iteritems():
        f = sum(campaign.duration*x[i][j] for (j, campaign) in campaigns.iteritems())
        cons.append( (f <= slot.length, 'slotlen_constr_%d' % i))
        
    for (j, campaign) in campaigns.iteritems():
        f = campaign.duration*campaign.cpvs*sum(campaign.reach[i]*x[i][j] for (i, slot) in slots.iteritems())
        cons.append( (f <= campaign.budget, 'budget_constr_%d' % j))
        
    for (j, campaign) in campaigns.iteritems():
        f = sum(x[i][j]*(campaign.reach[i]*campaign.cpvs - slot.price) for (i, slot) in slots.iteritems())
        cons.append( (f >= 0, 'at_least_price_sum_constr_%d' % j))
        
    prob = pu.LpProblem('wd', pu.LpMaximize)
    prob += sum(
        campaign.cpvs*campaign.duration*sum(
            x[i][j]*campaign.reach[i] 
            for (i, slot) in slots.iteritems()
        )
        for (j, campaign) in campaigns.iteritems()
    )
    # add constraints to problem
    for con in cons: prob += con
    return prob, (x,cons)

def solve(prob):
    status = prob.solve(solver)
#    print 'gap: ', prob.solver.epgap_actual
    return status

def generatePricesAndReaches(slots, campaigns, x):
    prices = {}
    reaches = {}
    for j, campaign in campaigns.iteritems():
        reach_total = sum(x[i][j].value()*campaign.reach[i] for (i, _slot) in slots.iteritems())
        price_total = reach_total*campaign.duration*campaign.cpvs
        reaches[j] = reach_total
        prices[j] = price_total
    return prices, reaches

def calcVcg(slots, campaigns, revenue_bid, prices_bid, prob, x):
    prices_vcg = {}
    for j in campaigns.iterkeys():
        coeffs = {}
        for i in slots.iterkeys():
            coeff = (prob.objective.get(x[i][j]))
            if coeff is not None:
                coeffs[i] = coeff
                prob.objective[x[i][j]] = 0
        status = solve(prob)
        revenue_vcg_single = pu.value(prob.objective)
        prices_vcg[j] = prices_bid[j] - (revenue_bid - revenue_vcg_single)
        
        for (i, coeff) in coeffs.iteritems():
            prob.objective[x[i][j]] = coeff
    return prices_vcg

slots = {
    1: Slot(1, 10.0, 10),
    2: Slot(2, 10.0, 30),
    4: Slot(3, 40.0, 30),
    5: Slot(3, 40.0, 30),
    6: Slot(3, 40.0, 30),
    7: Slot(3, 40.0, 30),
    8: Slot(3, 40.0, 30),
    9: Slot(3, 40.0, 30),
    11: Slot(3, 40.0, 30),
    12: Slot(3, 40.0, 30),
    13: Slot(3, 40.0, 30)
}

campaigns = {
    1: Campaign(1, 1.0, 10000.0, 10, {1:10, 2:20, 3:30, 4:10, 5:10, 6:12, 7:12, 8:13, 9:10, 10:10, 11:10, 12:10, 13:10}),
    2: Campaign(2, 1.0, 10000.0, 10, {1:30, 2:20, 3:10, 4:10, 5:10, 6:12, 7:12, 8:13, 9:10, 10:10, 11:10, 12:10, 13:10}),
    3: Campaign(3, 1.0, 10000.0, 10, {1:20, 2:20, 3:20, 4:10, 5:10, 6:12, 7:12, 8:13, 9:10, 10:10, 11:10, 12:10, 13:10}),
    4: Campaign(4, 10.0, 10000.0, 10, {1:10, 2:10, 3:10, 4:10, 5:10, 6:12, 7:12, 8:13, 9:10, 10:10, 11:10, 12:10, 13:10})
}
prob, (x, cons) = calc(slots, campaigns)

solve(prob)


prices_bid, reaches_bid = generatePricesAndReaches(slots, campaigns, x)
revenue_bid = pu.value(prob.objective)
print revenue_bid

print prices_bid
prices_vcg = calcVcg(slots, campaigns, revenue_bid, prices_bid, prob, x)
revenue_vcg = sum(prices_vcg.itervalues())
print revenue_vcg
print prices_vcg
        
objective_orig = prob.objective.copy()
winners = set(w for (w,price) in prices_bid.iteritems() if price)
prices_t = prices_vcg.copy()
prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)
pi = dict((w, pu.LpVariable('pi_%d' % w, cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=prices_bid[w])) for w in winners)
prob_ebpo += sum(pi_j for pi_j in pi.itervalues())

cnt = 0
while cnt < 1:
    cnt += 1
    for (j, campaign) in campaigns.iteritems():
        if not prices_bid[j]: continue
        ratio = prices_t[j]/prices_bid[j]
        for (i, slot) in slots.iteritems():
            try: prob.objective[x[i][j]] = objective_orig[x[i][j]]*ratio
            except KeyError: pass
    
    solve(prob)
    sep_value = pu.value(prob.objective)
    prices_sep, reaches_sep = generatePricesAndReaches(slots, campaigns, x)
    coalition_blocking = set(j for (j,price) in prices_sep.iteritems() if price)
    print sep_value
    print prices_sep
