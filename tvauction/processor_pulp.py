# -*- coding: utf-8; -*-

import sys
import logging

from collections import namedtuple, defaultdict
import pulp as pu
import math
import random

from pprint import pprint as pp

Slot = namedtuple('Slot', ('id','price','length'))
BidderInfo = namedtuple('BidderInfo', ('id','budget','length','attrib_min','attrib_values'))

class Gwd(object):
    def __init__(self,**kw):
        self.solver = pu.GUROBI(**kw)
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
        # add constraints to problem
        for con in cons: prob += con
        return (prob, (x,y,cons))
    
    def solve(self, prob, bidderInfos):
        logging.info('wdp:\tcalculating...')
        
        solver_status = prob.solve(self.solver)
        winners = frozenset(int(v.name[2:]) for v in prob.variables() if v.varValue==1 and v.name[:1]=='y')
        
        revenue_raw = pu.value(prob.objective)
        prices_raw = dict((w,bidderInfos[w].budget) for w in winners)
        
        logging.info('raw:\trevenue %d\tprices: %s' % (revenue_raw,sorted(prices_raw.iteritems())))
        return (solver_status,(revenue_raw, prices_raw, winners))
    
    def satisfyHighestBidders(self, amount_to_satisfy, slots, bidder_info_set, slots_time_remaining, prob_vars):
        # premise: try to look for an amount of bidders that have the highest impact on the system
        # and position them directly. this results, of course, in a suboptimal solution value.

        # no list side-effects
        slots_time_remaining = slots_time_remaining.copy()
        cons_highestbidders = []
        bidders_satisfied = []
        # sort the bidders by using their maximum value linearly and subtracting it from the price
        while amount_to_satisfy > 0:
            best_bidder_info = (0.0, None, None)
            for bidderInfo in bidder_info_set:
                budget_diff_info = self._getBudgetDiffInfo(bidderInfo, slots, slots_time_remaining)
                gain = 0.0
                attrib_needed = bidderInfo.attrib_min
                for (bdiff, attrib_value, slot) in budget_diff_info:
                    if bdiff <= 0 or attrib_needed <= 0: break
                    gain += bdiff*slot.length
                    attrib_needed -= attrib_value
                # the current best bidder causes the maximum gain while being satisfied
                if gain > best_bidder_info[0] and attrib_needed <= 0:
                    best_bidder_info = (gain, bidderInfo, budget_diff_info)
            
            # unpack best_bidder_info
            _gain, bidderInfo, budget_diff_info = best_bidder_info
            # if no bidder was found, break
            if bidderInfo is None: break
            # remove the bidderInfo from the list of bidderInfos and subtract amount_to_satisfy
            bidder_info_set = bidder_info_set-frozenset([bidderInfo])
            bidders_satisfied.append(bidderInfo)
            amount_to_satisfy -= 1
            # add constraints for the bidder.
            # this has to happen now (and not e.g. by using satisfySelectedBidders) because
            # the slots_time_remaining has to be updated for the next best bidder imeediately
            cons_bidder, slots_time_used_bidder = self._generateConstraintsForBudget(bidderInfo, budget_diff_info, prob_vars)
            for (slotid, time_used) in slots_time_used_bidder.iteritems():
                slots_time_remaining[slotid] -= time_used
        
        return cons_highestbidders, bidders_satisfied, slots_time_remaining
    
    def satisfySelectedBidders(self, slots, bidder_info_list, slots_time_remaining, prob_vars):
        cons_selectedbidders = []
        slots_time_remaining = slots_time_remaining.copy()
        for bidderInfo in bidder_info_list:
            budget_diff_info = self._getBudgetDiffInfo(bidderInfo, slots, slots_time_remaining)
            cons_bidder, slots_time_used_bidder = self._generateConstraintsForBudget(bidderInfo, budget_diff_info, prob_vars)
            for (slotid, time_used) in slots_time_used_bidder.iteritems():
                slots_time_remaining[slotid] -= time_used
            
            
    def _getBudgetDiffInfo(self, bidderInfo, slots, slots_time_remaining):
        budget_per_unit = (float(bidderInfo.budget) / bidderInfo.length) / bidderInfo.attrib_min
        # get the still available slots with the highest payoff
        budget_diff_info = sorted((
            (attrib_value*budget_per_unit-slot.price, attrib_value, slot)
            for (attrib_value, slot) in zip(bidderInfo.attrib_values, slots)
            if attrib_value > 0 and slots_time_remaining[slot.id] >= bidderInfo.length
        ),reverse=True)
        return budget_diff_info
    
    def _generateConstraintsForBudget(self, bidderInfo, budget_diff_info, prob_vars):
        x, _y, _cons = prob_vars
        cons = []
        slots_time_used = {}
        attrib_needed = bidderInfo.attrib_min
        # add winning constraints iteratively until the bidder is satisfied
        for (bdiff, attrib_value, slot) in budget_diff_info:
            if bdiff <= 0 or attrib_needed <= 0: break
            attrib_needed -= attrib_value
            con = x[slot.id][bidderInfo.id] == 1
            cons.append(con)
            slots_time_used[slot.id] = bidderInfo.length
        # add the y constraint (not really needed afaik because of big M constraints)
        # con = y[bidderInfo.id] == 1
        # cons.append(con)
        return cons, slots_time_used
 
        
        
class Vcg(object):
    def __init__(self, gwd):
        self.gwd = gwd
        pass
    def calculate(self, prob_gwd, prob_vars, bidderInfos, revenue_raw, winners, bidders_satisfied):
        logging.info('vcg:\tcalculating...')
        prices_vcg = {}
        for (step,w) in enumerate(winners):
            bidderInfoWinner = bidderInfos[w]
            revenue_without_w = self._calculate_step(prob_gwd, prob_vars, bidderInfoWinner)
            prices_vcg[w] = bidderInfoWinner.budget - (revenue_raw-revenue_without_w)
            logging.debug('calculating vcg - step %d of %d' % (step+1,len(winners)))
        revenue_vcg = sum(prices_vcg.itervalues())
        logging.info('vcg:\trevenue %d\tprices: %s' % (revenue_vcg,sorted(prices_vcg.iteritems())))
        return (revenue_vcg,prices_vcg)
    
    def _calculate_step(self, prob_gwd, prob_vars, bidderInfoWinner):
        x, y, _cons = prob_vars
        # remove all x and y constraints
        winner_ovars = frozenset([x_i[bidderInfoWinner.id] for x_i in x.itervalues()] + [y[bidderInfoWinner.id]])
        prob_vcg = prob_gwd.deepcopy()
        prob_vcg.name = 'vcg_%d' % bidderInfoWinner.id
        
        # remove vars from objective
        for ovar in prob_vcg.objective.keys():
            if ovar in winner_ovars: del prob_vcg.objective[ovar]
        # remove vars from constraints
        for conkey,con in prob_vcg.constraints.items():
            for ovar in con.keys():
                if ovar in winner_ovars: del prob_vcg.constraints[conkey][ovar]
            if not len(con): del prob_vcg.constraints[conkey]
        # solve the problem and get the revenue
        solver_status = prob_vcg.solve(self.gwd.solver)
        revenue_without_w = pu.value(prob_vcg.objective)
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
        pi = dict((w,pu.LpVariable('pi_%d' % (w,), cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=prices_raw[w])) for w in winners)
        m = pu.LpVariable('m',cat=pu.LpContinuous)
        # constants: ε (should be 'small enough')
        epsilon = sum(prices_vcg) - 1
        # ebpo objective function
        prob_ebpo += sum(pi.itervalues()) + epsilon*m
        # ebpo constraint: π_j - m <= π_j^vcg 
        for w in winners:
            prob_ebpo += pi[w]-m <= prices_vcg[w]
        ebpo_solver = pu.GUROBI(msg=False, mip=False)
        # abort after 1000 iterations (this should never happen)
        cnt = 0
        while cnt < 1000:
            cnt += 1
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
            solver_status = prob_sep.solve(self.gwd.solver)
            # save the value: z( π^t )
            revenue_sep = pu.value(prob_sep.objective)
            # check for a blocking coalition. if no coalition exists, break
            blocking_coalition_exists = revenue_sep > sum(prices_t.itervalues())
            if not blocking_coalition_exists: break
            # extend and solve the ebpo problem:
            blocking_coalition = frozenset(int(b.name[2:]) for b in prob_sep.variables() if b.varValue==1 and b.name[:1]=='y')
            logging.info('sep:\tblocking: %s' % (sorted(blocking_coalition),))
            
            winners_nonblocking = winners-blocking_coalition
            winners_blocking = winners&blocking_coalition
            # add (iteratively) new constraints to the ebpo problem
            prob_ebpo += sum(pi[wnb] for wnb in winners_nonblocking) >= revenue_sep - sum(prices_t[wb] for wb in winners_blocking)
            # solve the ebpo (this problem can be formulated as a continous LP).
            prob_ebpo.solve(ebpo_solver)
            # updated the π_t list
            prices_t = dict((int(b.name[3:]),b.varValue) for b in prob_ebpo.variables() if b.name[:2]=='pi')
        
        if cnt > 1000: raise Exception('too many iterations in core calculation')
        # there is no blocking coalition -> the current iteration of the ebpo contains core prices
        revenue_core = sum(prices_t.itervalues())
        logging.info('core:\trevenue %d\tprices: %s' % (revenue_core,prices_t.items()))
        return (revenue_core, prices_t)


def solve(slots, bidderInfos, timeLimit=10, **kw):

    # vars needed for the heuristic    
    bidder_info_set = frozenset(bidderInfos)
    bidders_satisfied = []
    slots_time_remaining = dict((slot.id,slot.length) for slot in slots)
    
    # generate the gwd    
    gwd = Gwd(msg=True)
    prob_gwd, prob_vars = gwd.generate(slots, bidderInfos)
    
    # solve the gwd with a time limit to determine whether a heuristic has to be used
    gwd.solver.timeLimit = timeLimit
    amount_to_satisfy = 0
    
#    solver_status, (revenue_raw, prices_raw, winners) = gwd.solve(prob_gwd, bidderInfos)
    solver_status = pu.LpStatusNotSolved
    while solver_status != pu.LpStatusOptimal:
        amount_to_satisfy += round(len(bidderInfos)*0.1)
        if amount_to_satisfy >= len(bidderInfos): 
            raise Exception('problem too big to solve')
        cons_highest, bidders_now_satisfied, slots_time_remaining = gwd.satisfyHighestBidders(amount_to_satisfy, slots, bidder_info_set, slots_time_remaining, prob_vars)
        bidder_info_set -= frozenset(bidders_now_satisfied)
        bidders_satisfied.extend(bidders_now_satisfied)
        for con in cons_highest: 
            prob_gwd += con
        solver_status, (revenue_raw, prices_raw, winners) = gwd.solve(prob_gwd, bidderInfos)
    
    # reset timeLimit and epgap. crucial for the correctness of the sep problem
    gwd.solver.timeLimit = None
    gwd.solver.epgap = None
    
    #
    vcg = Vcg(gwd)
    revenue_vcg, prices_vcg = vcg.calculate(prob_gwd, prob_vars, bidderInfos, revenue_raw, winners, bidders_satisfied)
    
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
        raise ValueError('Invalid log level: %s' % options.loglevel)
    if not options.scenario:
        raise ValueError('Scenario needed')
    
    logging.basicConfig(level=numeric_level)
    
    slots, bidderInfos = json.loads(options.scenario)
    res = solve(slots, bidderInfos)
    
    print json.dumps(res,indent=2)    
