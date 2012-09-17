# -*- coding: utf-8; -*-

import sys
import logging

from collections import namedtuple, defaultdict
import pulp as pu
import math
import random

from pprint import pprint as pp

SOLVER_MSG = False

Slot = namedtuple('Slot', ('id','price','length'))
BidderInfo = namedtuple('BidderInfo', ('id','budget','length','attrib_min','attrib_values'))

class Gwd(object):
    def __init__(self):
        self.solver = pu.GUROBI(msg=SOLVER_MSG)
        self.prob = None
        
    def generate(self, slots, bidderInfos):
        '''the winner determination, implemented as a multiple knapsack problem'''
        
        # x determines whether a bidder can air in a certain slot
        x = pu.LpVariable.dicts('x', (slots.keys(),bidderInfos.keys()), cat=pu.LpBinary)
        # y determines whether a winner has won
        y = pu.LpVariable.dicts('y', (bidderInfos.keys(),), cat=pu.LpBinary)
        # initialize constraints
        cons = []
        # the sum of all assigned ad lengths has to be at most the length of the slot
        for (i,slot) in slots.iteritems():
            f = sum(bidderInfo.length*x[i][j] for (j,bidderInfo) in bidderInfos.iteritems())
            cons.append(f <= slot.length)
        # match the bidders' demands regarding their attributes
        # attrib_values has to be a list with the same length
        # as the slots list
        for (j,bidderInfo) in bidderInfos.iteritems():
            assert len(slots) == len(bidderInfo.attrib_values)
            M = sum(bidderInfo.attrib_values.itervalues())+1
            f = sum(attrib_value*x[i][j] for (i,attrib_value) in bidderInfo.attrib_values.iteritems())
            f2 = bidderInfo.attrib_min - f
            cons.append(f <= M*y[j])
            cons.append(f2 <= M*(1-y[j]))
        # user can at most spend the maximum price
        for (j,bidderInfo) in bidderInfos.iteritems():
            f = sum(bidderInfo.length*slot.price*x[i][j] for (i,slot) in slots.iteritems())
            cons.append(f <= bidderInfo.budget)
        # oovar domain=bool takes already care of min and max bounds
        prob = pu.LpProblem('gwd', pu.LpMaximize)
        prob += sum(bidderInfo.budget*y[j] for (j,bidderInfo) in bidderInfos.iteritems())
        # add constraints to problem
        for con in cons: prob += con
        return (prob, (x,y,cons))
    
    def solve(self, prob, bidderInfos, prob_vars):
        logging.info('wdp:\tcalculating...')
        
        solver_status = prob.solve(self.solver)
        _x, y, _cons = prob_vars
        logging.info('wdp:\tstatus: %s' % pu.LpStatus[solver_status])
        winners = frozenset(j for (j,y_j) in y.iteritems() if y_j.varValue==1)
        
        revenue_raw = pu.value(prob.objective)
        prices_raw = dict((w,bidderInfos[w].budget) for w in winners)
        
        logging.info('raw:\trevenue %d\tprices: %s' % (revenue_raw,sorted(prices_raw.iteritems())))
        return (solver_status,(revenue_raw, prices_raw, winners))
    
class ReservePrice(object):
    '''checks if all winners have to pay at least the reserve price.
    if this is not the case, their price is changed accordingly'''
    def solve(self, slots, bidderInfos, winners_slots, prices_before):
        prices_after = {}
        for w,slot_ids_won in winners_slots.iteritems():
            bidderInfo = bidderInfos[w]
            price_reserve = sum(slots[slot_id].price for slot_id in slot_ids_won)*bidderInfo.length
            prices_after[w] = max(price_reserve, prices_before[w])
        revenue_after = sum(prices_after.itervalues())
        return (revenue_after,prices_after)

class VcgFake(object):
    def __init__(self, gwd):
        self.gwd = gwd
    def solve(self, slots, bidderInfos, revenue_raw, winners, prob_gwd, prob_vars):
        prices_zero = dict((w,0) for w in winners)
        return (0, prices_zero)

class Vcg(object):
    def __init__(self, gwd):
        self.gwd = gwd
    
    def solve(self, slots, bidderInfos, revenue_raw, winners, prob_gwd, prob_vars):
        logging.info('vcg:\tcalculating...')
        prices_vcg = {}
        prob_vcg = prob_gwd.deepcopy()
        for w in winners:
            bidderinfo_winner = bidderInfos[w]
            revenue_without_w = self._solve_step(prob_vcg, prob_vars, w)
            prices_vcg[w] = max(0, bidderinfo_winner.budget - max(0,(revenue_raw-revenue_without_w)))
        revenue_vcg = sum(prices_vcg.itervalues())
        logging.info('vcg:\trevenue %d\tprices: %s' % (revenue_vcg,sorted(prices_vcg.iteritems())))
        return (revenue_vcg,prices_vcg)
    
    def _solve_step(self, prob_vcg, prob_vars, winner_id):
        '''takes the original gwd problem, and forces a winner to lose. used in vcg calculation'''
        _x, y, _cons = prob_vars
        
        logging.info('vcg:\tcalculating - without winner %s' % (winner_id,))
        prob_vcg.name = 'vcg_%d' % winner_id
        prob_vcg.addConstraint(y[winner_id] == 0, 'vcg_%d' % winner_id)
        
        solver_status = prob_vcg.solve(self.gwd.solver)
        logging.info('vcg:\tstatus: %s' % pu.LpStatus[solver_status])
        del prob_vcg.constraints['vcg_%d' % winner_id]
        
        revenue_without_w = pu.value(prob_vcg.objective)
        return revenue_without_w
        
class CorePricing(object):
    def __init__(self, gwd):
        self.gwd = gwd
    
    def solve(self, prob_gwd, bidderInfos, winners, prices_raw, prices_vcg):
        # build ebpo
        prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)
        prices_t = prices_vcg.copy()
        
        # variables: π_j, m
        pi = dict((w,pu.LpVariable('pi_%d' % (w,), cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=prices_raw[w])) for w in winners)
        m = pu.LpVariable('m',cat=pu.LpContinuous)
        
        # constants: ε (should be 'small enough')
        epsilon = 1
        
        # ebpo objective function
        prob_ebpo += sum(pi.itervalues()) + epsilon*m
        
        # ebpo constraint: π_j - m <= π_j^vcg 
        for w in winners:
            prob_ebpo += pi[w]-m <= prices_vcg[w]
        ebpo_solver = pu.GUROBI(msg=SOLVER_MSG, mip=False)
        
        # initialize revenue_sep and prices_t_sum vars, used for comparisons
        revenue_sep = revenue_sep_last = 0
        prices_t_sum = prices_t_sum_last = sum(prices_t.itervalues())
        
        # abort after 1000 iterations (this should never happen)
        for cnt in xrange(1000):
            # make a copy of prob_gwd, since we are using it as basis
            prob_sep = prob_gwd.deepcopy()
            prob_sep.name = 'sep_%d' % cnt
            # build sep t variable
            t = dict((w,pu.LpVariable('t_%d' % (w,), cat=pu.LpBinary)) for w in winners)
            # add the 'maximum coalition contribution' to the objective fn
            prob_sep.objective -= sum((prices_raw[w]-prices_t[w])*t[w] for w in winners)
            # add all sep constraints, setting y_i <= t_i
            for w in winners:
                prob_sep += prob_sep.variablesDict()['y_%d' % w] <= t[w]
            # solve it
            logging.info('sep:\tcalculating - step %d' % cnt)
            solver_status = prob_sep.solve(self.gwd.solver)
            logging.info('sep:\tstatus: %s' % pu.LpStatus[solver_status])
            
            # save the value: z(π^t)
            revenue_sep, revenue_sep_last = pu.value(prob_sep.objective), revenue_sep
            
            # check for a blocking coalition. if no coalition exists, break 
            blocking_coalition_exists = revenue_sep > sum(prices_t.itervalues())
            if not blocking_coalition_exists: 
                logging.info('sep:\tvalue: %d, blocking: None' % revenue_sep)
                break

            # get the blocking coalition
            blocking_coalition = frozenset(int(b.name[2:]) for b in prob_sep.variables() if b.varValue==1 and b.name[:1]=='y')
            logging.info('sep:\tvalue: %d, blocking: %s' % (revenue_sep, sorted(blocking_coalition),))
            
            revenue_blocking_coalition = sum(b.budget for (b_id,b) in bidderInfos.iteritems() if b_id in blocking_coalition)
            logging.info('sep:\tvalue_blocking_coalition: %d, bigger: %s' % (revenue_blocking_coalition, revenue_blocking_coalition > sum(prices_raw.itervalues())))
            
            winners_nonblocking = winners-blocking_coalition
            winners_blocking = winners&blocking_coalition
            
            # add (iteratively) new constraints to the ebpo problem.
            # revenue_without_blocking can be at most the sum of the prices_raw of winners_nonblocking
            revenue_without_blocking = revenue_sep - sum(prices_t[wb] for wb in winners_blocking)
            revenue_without_blocking = min(sum(prices_raw[wnb] for wnb in winners_nonblocking), revenue_without_blocking)
            
            prob_ebpo += sum(pi[wnb] for wnb in winners_nonblocking) >= revenue_without_blocking
            
            # solve the ebpo (this problem can be formulated as a continuous LP).
            prob_ebpo.writeLP('ebpo.lp')
            logging.info('ebpo:\tcalculating - step %s' % (cnt),)
            solver_status = prob_ebpo.solve(ebpo_solver)
            assert solver_status == pu.LpStatusOptimal
            
            # update the π_t list. sum(π_t) has to be equal or increase for each t
            prices_t = dict((int(b.name[3:]), b.varValue) for b in prob_ebpo.variables() if b.name[:2]=='pi')
            prices_t_sum, prices_t_sum_last = sum(prices_t.itervalues()), prices_t_sum
            logging.info('ebpo:\trevenue: %d' % prices_t_sum)

            # if both z(π^t) == z(π^t-1) and θ^t == θ^t-1, we are in a local optimum we cannot escape
            # proposition: this happens when the gwd returned a suboptimal solution, causing a winner allocation.
            if revenue_sep == revenue_sep_last and prices_t_sum == prices_t_sum_last:
                logging.warn('core:\tvalue did not change. aborting.') 
                break
        else:
            logging.warn('core:\ttoo many iterations in core calculation. aborting.')
            
        # there is no blocking coalition -> the current iteration of the ebpo contains core prices
        revenue_core = sum(prices_t.itervalues())
        logging.info('core:\trevenue %d\tprices: %s' % (revenue_core,[(k,round(v,2)) for (k,v) in prices_t.iteritems()]))
        return (revenue_core, prices_t)

class TvAuctionProcessor(object):
    def __init__(self):
        self.gwdClass = Gwd
        self.vcgClass = Vcg
        self.reservePriceClass = ReservePrice
        self.coreClass = CorePricing
        
    def isOptimal(self,solver_status):
        return solver_status == pu.LpStatusOptimal
    
    def solve(self, slots, bidderInfos, timeLimit=20, epgap=0.02):
        '''solve the wdp and pricing problem.
        
        @param slots:       a dict of Slot objects
        @param bidderInfos: a dict of BidderInfo objects
        @param timeLimit:   int|null, if int, then the time will be doubled for the gwd. is used for all integer problems.
        @param epgap:       float|null, if int, is used for all integer problems.'''
        
        # generate the gwd    
        gwd = self.gwdClass()
        prob_gwd, prob_vars = gwd.generate(slots, bidderInfos)
        
        # add a gap and timelimit if set.
        # we double the timelimit for the gwd.
        if timeLimit is not None: gwd.solver.timeLimit = timeLimit * 2
        if epgap is not None: gwd.solver.epgap = epgap
        
        _solver_status, (revenue_raw, prices_raw, winners) = gwd.solve(prob_gwd, bidderInfos, prob_vars)
        
        # get the slots for the winners
        x, _y, _cons = prob_vars
        slots_winners = {}
        winners_slots = dict((w,[]) for w in winners)
        for slot_id, slot_user_vars in x.iteritems():
            slot_winners = [user_id for (user_id,has_won) in slot_user_vars.iteritems() if round(has_won.value()) == 1]
            for slot_winner in slot_winners: winners_slots[slot_winner].append(slot_id)
            slots_winners[slot_id] = slot_winners

        # if timelimit was set: adjust it
        if timeLimit is not None: gwd.solver.timeLimit = timeLimit
        
        # solve vcg
        vcg = self.vcgClass(gwd)
        _revenue_vcg, prices_vcg = vcg.solve(slots, bidderInfos, revenue_raw, winners, prob_gwd, prob_vars)

        # solve core pricing problem
        core = self.coreClass(gwd)
        _revenue_core, prices_core = core.solve(prob_gwd, bidderInfos, winners, prices_raw, prices_vcg)
        
        # raise prices to the reserve price if needed
        reservePrice = self.reservePriceClass()
        _revenue_after, prices_after = reservePrice.solve(slots, bidderInfos, winners_slots, prices_core)
        
        return {
            'winners': list(winners),
            'slots_winners': slots_winners,
            'prices_raw': prices_raw,
            'prices_vcg': prices_vcg,
            'prices_core': prices_core,
            'prices_final': prices_after
        }
     
if __name__ == '__main__':
    import json
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-l','--log', dest='loglevel', help='the log level', default='WARN')
    parser.add_option('-s','--scenario', dest='scenario', help='the scenario')
    options = parser.parse_args()[0]
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)
    if not options.scenario:
        options.scenario = '[{"1":{"id":1,"price":1.0,"length":120}},{"1":{"id":1,"budget":100,"length":20,"attrib_min":1,"attrib_values":{"1":1}},"2":{"id":2,"budget":100,"length":10,"attrib_min":1,"attrib_values":{"1":1}}}]'
#        raise ValueError('Scenario needed')
    
    logging.basicConfig(level=numeric_level)
    
    slots_imported, bidderInfos_imported = json.loads(options.scenario)
    slots = dict( (s['id'],Slot(**s)) for s in slots_imported.itervalues() )
    bidderInfos = dict( (b['id'],BidderInfo(**b)) for b in bidderInfos_imported.itervalues() )
    for bidderInfo in bidderInfos.itervalues():
        attrib_values = bidderInfo.attrib_values
        for av in attrib_values.keys():
            attrib_values[int(av)] = attrib_values.pop(av)
    proc = TvAuctionProcessor()
    res = proc.solve(slots, bidderInfos)

    print json.dumps(res,indent=2)    
