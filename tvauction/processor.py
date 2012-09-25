# -*- coding: utf-8; -*-

from collections import namedtuple, defaultdict
from pprint import pprint as pp
import logging
import math
import pulp as pu
import random
import sys


UPDATE_EBPO = True
SOLVER_MSG = False
SOLVER_CLASS = pu.GUROBI

Slot = namedtuple('Slot', ('id','price','length'))
BidderInfo = namedtuple('BidderInfo', ('id','budget','length','attrib_min','attrib_values'))

class Gwd(object):
    '''general winner determination'''
    def __init__(self):
        self.solver = SOLVER_CLASS(msg=SOLVER_MSG)
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
            cons.append( (f <= slot.length , 'slotlen_constr_%d' % i) )
        # match the bidders' demands regarding their attributes
        # attrib_values has to be a list with the same length
        # as the slots list
        for (j,bidderInfo) in bidderInfos.iteritems():
            assert len(slots) == len(bidderInfo.attrib_values)
            M = sum(bidderInfo.attrib_values.itervalues())+1
            f = sum(attrib_value*x[i][j] for (i,attrib_value) in bidderInfo.attrib_values.iteritems())
            f2 = bidderInfo.attrib_min - f
            cons.append( (f <= M*y[j], 'M_constr_%d.1' % j) )
            cons.append( (f2 <= M*(1-y[j]), 'M_constr_%d.2' % j) )
        # user can at most spend the maximum price
        for (j,bidderInfo) in bidderInfos.iteritems():
            f = sum(bidderInfo.length*slot.price*x[i][j] for (i,slot) in slots.iteritems())
            cons.append( (f <= bidderInfo.budget, 'budget_constr_%d' % j) )
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
        winners = frozenset(j for (j,y_j) in y.iteritems() if round(y_j.value())==1)
        
        revenue_raw = pu.value(prob.objective)
        prices_raw = dict((w,bidderInfos[w].budget) for w in winners)
        
        logging.info('raw:\trevenue %d, prices: %s' % (revenue_raw,sorted(prices_raw.iteritems())))
        return (solver_status,(revenue_raw, prices_raw, winners))
    
    def getSlotAssignments(self, winners, x):
        winners_slots = dict((w,[]) for w in winners)
        for slot_id, slot_user_vars in x.iteritems():
            slot_winners = [user_id for (user_id,has_won) in slot_user_vars.iteritems() if user_id in winners and has_won.value() and round(has_won.value()) == 1]
            for slot_winner in slot_winners: winners_slots[slot_winner].append(slot_id)
        return winners_slots
    
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

#
# TODO: use the epgap to raise the revenue_without_bidder (so vcg price gets lower)
# if not, we risk too high vcg prices which could cause infeasibilities!
#
class InitialPricing(object):
    def __init__(self, gwd):
        self.gwd = gwd
        
    def getCoalitions(self, slots, bidderInfos, revenue_raw, winners, prob_gwd, prob_vars):
        logging.info('vcg:\tcalculating...')
        winners_without_bidders = {}
        prob_vcg = prob_gwd.deepcopy()
        for w in winners:
            winners_without_bidders[w] = self.solveStep(prob_vcg, prob_vars, w)
        return winners_without_bidders
        
    def getPricesForBidders(self, bidderInfos, revenue_raw, winners, winners_without_bidders):
        res = {}
        for w in winners:
            revenue_without_bidder = sum(bidderInfos[wo].budget for wo in winners_without_bidders[w])
            res[w] = self.getPriceForBidder(bidderInfos[w].budget, revenue_raw, revenue_without_bidder)
        return res
        
class Zero(InitialPricing):
    '''the zero pricing vector class'''
        
    def solveStep(self, prob_vcg, prob_vars, winner_id):
        return frozenset()
    
    def getPriceForBidder(self, budget, revenue_raw, revenue_without_bidder):
        return 0
    
class Vcg(InitialPricing):
    def __init__(self, gwd):
        self.gwd = gwd
    
    def solveStep(self, prob_vcg, prob_vars, winner_id):
        '''takes the original gwd problem, and forces a winner to lose. used in vcg calculation'''
        _x, y, _cons = prob_vars
        
        logging.info('vcg:\tcalculating - without winner %s' % (winner_id,))
        prob_vcg.name = 'vcg_%d' % winner_id
        prob_vcg.addConstraint(y[winner_id] == 0, 'vcg_%d' % winner_id)
        
        solver_status = prob_vcg.solve(self.gwd.solver)
        logging.info('vcg:\tstatus: %s' % pu.LpStatus[solver_status])
        del prob_vcg.constraints['vcg_%d' % winner_id]
        winners_without_w = frozenset(j for (j,y_j) in y.iteritems() if round(y_j.value())==1)
        return winners_without_w
    
    def getPriceForBidder(self, budget, revenue_raw, revenue_without_bidder):
        return max(0, budget - max(0, (revenue_raw-revenue_without_bidder)))
    
class CorePricing(object):
    
    TRIM_VALUES=1
    SWITCH_COALITIONS=2
    REUSE_COALITIONS=3
    
    def __init__(self, gwd, vcg, algorithm=REUSE_COALITIONS):
        self.gwd = gwd
        self.vcg = vcg
        self.algorithm = algorithm
    
    
    def _createSep(self, name, winners, bidderInfos, prices_t, prob_gwd, y):
        # make a copy of prob_gwd, since we are using it as basis
        prob_sep = prob_gwd.deepcopy()
        prob_sep.name = name
        # build sep t variable and modify objective
        t = dict((w,pu.LpVariable('t_%d' % (w,), cat=pu.LpBinary)) for w in winners)
        prob_sep.objective -= sum((bidderInfos[w].budget-prices_t[w])*t[w] for w in winners)
        # add all sep constraints, setting y_i <= t_i
        for w in winners: prob_sep += y[w] <= t[w]
        return prob_sep
    
    def _createEbpo(self, name, winners, bidderInfos, coalitions, prices_vcg):
        # init ebpo
        prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)
        prob_ebpo.name = name
        # constants: ε (should be 'small enough')
        epsilon = 1e-31
        # variables: m, π_j 
        m = pu.LpVariable('m',cat=pu.LpContinuous)
        pi = dict((w,pu.LpVariable('pi_%d' % (w,), cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=bidderInfos[w].budget)) for w in winners)
        # add objective function
        prob_ebpo += sum(pi.itervalues()) + epsilon*m
        # add pi constraints
        for w in winners: prob_ebpo += (pi[w]-m <= prices_vcg[w], 'pi_constr_%d' % w)
        # add coalition constraints for all coalitions without the winning coalition TODO only to it if updating
        for c in coalitions-{winners}: prob_ebpo += sum(pi[wnb] for wnb in winners-c) >= sum(bidderInfos[j].budget for j in c-winners)
        return (prob_ebpo, pi)
    
    def _updateToBestCoalition(self, coalitions, winners_without_bidders, current_coalition, bidderInfos, prob_vcg, prob_vars):
        '''iteratively update to the best coalition, generating new coalitions while getting vcg coalitions on the way.
        update coalitions and winners_without_bidders in-place!'''
        coalition_changed = False
        best_coalition = current_coalition
        while best_coalition != self._getBestCoalition(coalitions, bidderInfos):
            coalition_changed = True
            best_coalition = self._getBestCoalition(coalitions, bidderInfos)
            winners_without_bidders.update(
                (j, self.vcg.solveStep(prob_vcg,prob_vars,j)) 
                for j in best_coalition if j not in winners_without_bidders
            )
            coalitions.update(winners_without_bidders.itervalues())
        return coalition_changed, best_coalition
    
    @staticmethod
    def _getBestCoalition(coalitions, bidderInfos):
        return max(coalitions,key=lambda c: sum(bidderInfos[j].budget for j in c))
    
    def solve(self, prob_gwd, bidderInfos, winners, winners_without_bidders, prob_vars):
        # init coalition
        coalitions = set()
        
        # we need a prob_vcg because of the generation of vcg prices for new coalition entries
        prob_vcg = prob_gwd.copy()
        
        # init ebpo and variables
        prob_ebpo = None
        ebpo_solver = SOLVER_CLASS(msg=SOLVER_MSG, mip=False)
        pi = {}
        
        # add winners and winners_without_bidders to the coalitions
        coalitions.add(winners)
        coalitions.update(winners_without_bidders.itervalues())
        
        # get currently highest valued coalition
        _coalition_changed, winners = self._updateToBestCoalition(coalitions, winners_without_bidders, winners, bidderInfos, prob_vcg, prob_vars)        
        revenue_raw = sum(bidderInfos[w].budget for w in winners)
        
        # init prices_vcg and prices_t
        prices_vcg = self.vcg.getPricesForBidders(bidderInfos, revenue_raw, winners, winners_without_bidders)
        prices_t = prices_vcg.copy()
        
        # initialize obj_value_sep and prices_t_sum vars, used for comparisons
        obj_value_sep = obj_value_sep_last = 0
        prices_t_sum = prices_t_sum_last = sum(prices_t.itervalues())
        blocking_coalition = blocking_coalition_last = None
        
        # store information about the steps (in order to get insights / draw graphs of the process)
        step_info = [{'raw':revenue_raw,'vcg':sum(prices_vcg.itervalues())}]
        
        # get problem vars
        _x, y, _cons = prob_vars
        
        # abort after 1000 iterations (this should never happen)
        for cnt in xrange(1000):
            # create sep
            prob_sep = self._createSep('sep_%d' % cnt, winners, bidderInfos, prices_t, prob_gwd, y)
            
            # solve it
            logging.info('sep:\tcalculating - step %d' % cnt)
            solver_status = prob_sep.solve(self.gwd.solver)
            logging.info('sep:\tstatus: %s' % pu.LpStatus[solver_status])
            
            # save the value: z(π^t)
            obj_value_sep, obj_value_sep_last = pu.value(prob_sep.objective), obj_value_sep
            
            # check for no blocking coalition exists. if yes, break
            if not obj_value_sep > sum(prices_t.itervalues()):
                logging.info('sep:\tvalue: %d, blocking: None' % obj_value_sep)
                break
            
            # get the blocking coalition
            blocking_coalition, blocking_coalition_last = frozenset(bidder_id for (bidder_id,y_j) in y.iteritems() if round(y_j.value())==1), blocking_coalition
            logging.info('sep:\tvalue: %d, blocking: %s' % (obj_value_sep, sorted(blocking_coalition),))
            # and add it to the list of coalitions
            coalitions.add(blocking_coalition)
            
            # if the added coalition is better than the currently winning coalition, get all new vcg coalitions and repeat
            winners_changed, winners = self._updateToBestCoalition(coalitions, winners_without_bidders, winners, bidderInfos, prob_vcg, prob_vars)
            
            if prob_ebpo is None or winners_changed:
                revenue_raw = sum(bidderInfos[w].budget for w in winners)
                prices_vcg = self.vcg.getPricesForBidders(bidderInfos, revenue_raw, winners, winners_without_bidders)
                prob_ebpo, pi = self._createEbpo('ebpo_%d' % cnt, winners, bidderInfos, coalitions, prices_vcg)
            else:
                prob_ebpo += sum(pi[wnb] for wnb in winners-blocking_coalition) >= sum(bidderInfos[j].budget for j in blocking_coalition-winners)
                
            # ebpo: solve (this problem can be formulated as a continuous LP).
            logging.info('ebpo:\tcalculating - step %s' % (cnt),)
            solver_status = prob_ebpo.solve(ebpo_solver)
            
            # update the π_t list. sum(π_t) has to be equal or increase for each t
            prices_t = dict((j,pi_j.value() or pi_j.lowBound) for (j,pi_j) in pi.iteritems())
            prices_t_sum, prices_t_sum_last = round(sum(prices_t.itervalues()),2), prices_t_sum
            # ebpo: has to be feasible
            assert solver_status == pu.LpStatusOptimal, 'ebpo: not optimal - %s' % pu.LpStatus[solver_status]
            
            # update step_info
            step_info.append({'ebpo':prices_t_sum,'sep':obj_value_sep,'raw':revenue_raw,'vcg':sum(prices_vcg.itervalues())})
            
            # if both z(π^t) == z(π^t-1) and θ^t == θ^t-1, we are in a local optimum we cannot escape
            # proposition: this happens when the gwd returned a suboptimal solution, causing a winner allocation.
            if obj_value_sep == obj_value_sep_last and prices_t_sum == prices_t_sum_last and blocking_coalition == blocking_coalition_last:
                logging.warn('core:\tvalue did not change. aborting.') 
                break
        else:
            logging.warn('core:\ttoo many iterations in core calculation. aborting.')
            
        revenue_core = sum(prices_t.itervalues())
        logging.info('core:\trevenue %d, prices: %s' % ( revenue_core, sorted((j,round(pi_j,2)) for (j,pi_j) in prices_t.iteritems()) ) )
        return (winners, prices_t, prices_vcg, step_info)
        
class TvAuctionProcessor(object):
    def __init__(self):
        self.gwdClass = Gwd
        self.vcgClass = Vcg
        self.reservePriceClass = ReservePrice
        self.coreClass = CorePricing
        self.core_algorithm = CorePricing.REUSE_COALITIONS
        
    def isOptimal(self,solver_status):
        return solver_status == pu.LpStatusOptimal
    
    def solve(self, slots, bidderInfos, timeLimit=20, timeLimitGwd=20, epgap=None):
        '''solve the wdp and pricing problem.
        
        @param slots:        a dict of Slot objects
        @param bidderInfos:  a dict of BidderInfo objects
        @param timeLimit:    int|null, sets the time limit for all integer problems.
        @param timeLimitGwd: int|null, lets you override the time limit for the gwd because of its importance. 
        @param epgap:        float|null, is used for all integer problems.'''
        
        # generate the gwd    
        gwd = self.gwdClass()
        prob_gwd, prob_vars = gwd.generate(slots, bidderInfos)
        
        x, y, _cons = prob_vars
        
        gwd.solver.epgap = epgap
        gwd.solver.timeLimit = timeLimitGwd
        
        _solver_status, (revenue_raw, prices_raw, winners) = gwd.solve(prob_gwd, bidderInfos, prob_vars)
        
        # if timelimit was set: adjust it
        gwd.solver.timeLimit = timeLimit
        
        # solve vcg
        vcg = self.vcgClass(gwd)
        winners_without_bidders = vcg.getCoalitions(slots, bidderInfos, revenue_raw, winners, prob_gwd, prob_vars)

        # solve core pricing problem
        core = self.coreClass(gwd, vcg, self.core_algorithm)
        winners_core, prices_core, prices_vcg, step_info = core.solve(prob_gwd, bidderInfos, winners, winners_without_bidders, prob_vars)
        
        # re-solve the gwd to get the slot assignments (this is ok, since we pre-seed the prob with constraints)
        if winners_core != winners:
            for wc in winners_core: prob_gwd += y[wc] == 1
            gwd.solver.epgap = epgap
            gwd.solver.timeLimit = timeLimitGwd
            _solver_status, (revenue_raw, prices_raw, winners) = gwd.solve(prob_gwd, bidderInfos, prob_vars)
            
        # get the slots for the winners
        winners_slots = gwd.getSlotAssignments(winners, x)

        # raise prices to the reserve price if needed
        reservePrice = self.reservePriceClass()
        _revenue_after, prices_after = reservePrice.solve(slots, bidderInfos, winners_slots, prices_core)
        
        return {
            'winners': sorted(winners_slots.keys()),
            'winners_slots': winners_slots,
            'prices_raw': prices_raw,
            'prices_vcg': prices_vcg,
            'prices_core': prices_core,
            'prices_final': prices_after,
            'step_info': step_info
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
