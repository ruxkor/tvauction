# -*- coding: utf-8; -*-

import sys
import logging

from collections import namedtuple, defaultdict
import pulp as pu
import math
import random

from pprint import pprint as pp

SOLVER_MSG = False
SOLVER_CLASS = pu.GUROBI

Slot = namedtuple('Slot', ('id','price','length'))
BidderInfo = namedtuple('BidderInfo', ('id','budget','length','attrib_min','attrib_values'))

class Gwd(object):
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
        
        logging.info('raw:\trevenue %d\tprices: %s' % (revenue_raw,sorted(prices_raw.iteritems())))
        return (solver_status,(revenue_raw, prices_raw, winners))
    
    def getSlotAssignments(self, winners, x):
        winners_slots = dict((w,[]) for w in winners)
        for slot_id, slot_user_vars in x.iteritems():
            slot_winners = [user_id for (user_id,has_won) in slot_user_vars.iteritems() if has_won.value() and round(has_won.value()) == 1]
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
class VcgFake(object):
    def __init__(self, gwd):
        self.gwd = gwd
        
    def solve(self, slots, bidderInfos, revenue_raw, winners, prob_gwd, prob_vars):
        prices_zero = dict((w,0) for w in winners)
        return (0, prices_zero, prices_zero)
    
    def solveStep(self, prob_vcg, prob_vars, winner_id):
        return 0
    
    def getPriceForBidder(self, budget, revenue_raw, revenue_without_bidder):
        return 0
    
    def getPricesForBidders(self, bidderInfos, revenue_raw, revenues_without_bidders):
        return dict(
            (w, self.getPriceForBidder(bidderInfos[w].budget, revenue_raw, revenue_without_w))
            for (w,revenue_without_w) in revenues_without_bidders.iteritems()
        )
    

class Vcg(object):
    def __init__(self, gwd):
        self.gwd = gwd
    
    def solve(self, slots, bidderInfos, revenue_raw, winners, prob_gwd, prob_vars):
        logging.info('vcg:\tcalculating...')
        prices_vcg = {}
        revenues_without_bidders = {}
        prob_vcg = prob_gwd.deepcopy()
        for w in winners:
            revenues_without_bidders[w] = self.solveStep(prob_vcg, prob_vars, w)
            prices_vcg[w] = self.getPriceForBidder(bidderInfos[w].budget, revenue_raw, revenues_without_bidders[w])
        revenue_vcg = sum(prices_vcg.itervalues())
        logging.info('vcg:\trevenue %d\tprices: %s' % (revenue_vcg,sorted(prices_vcg.iteritems())))
        return (revenue_vcg, prices_vcg, revenues_without_bidders)
    
    def solveStep(self, prob_vcg, prob_vars, winner_id):
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
    
    def getPriceForBidder(self, budget, revenue_raw, revenue_without_bidder):
        return max(0, budget - max(0, (revenue_raw-revenue_without_bidder)))
    
    def getPricesForBidders(self, bidderInfos, revenue_raw, revenues_without_bidders):
        return dict(
            (w, self.getPriceForBidder(bidderInfos[w].budget, revenue_raw, revenue_without_w))
            for (w,revenue_without_w) in revenues_without_bidders.iteritems()
        )
        
        
class CorePricing(object):
    
    TRIM_VALUES=1
    SWITCH_COALITION=2
    
    def __init__(self, gwd, vcg, algorithm=SWITCH_COALITION):
        self.gwd = gwd
        self.vcg = vcg
        self.algorithm = algorithm
    
    def solve(self, prob_gwd, bidderInfos, winners, prices_raw, revenues_without_bidders, prob_vars):
        # copy some variables in order to not cause side effects in them
        prices_raw = prices_raw.copy()
        revenues_without_bidders = revenues_without_bidders.copy()
        
        # init winners_slots, needed if the coalition gets changed
        winners_slots = None
        
        revenue_raw = sum(prices_raw.itervalues())
        prices_vcg = self.vcg.getPricesForBidders(bidderInfos, revenue_raw, revenues_without_bidders)
        
        # build ebpo
        prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)
        prices_t = prices_vcg.copy()
        
        # constants: ε (should be 'small enough')
        epsilon = 1e-31
        
        # variables: π_j, m
        # π_j is built for all bidders (instead of only for the winners), since the winning coalition may change 
        m = pu.LpVariable('m',cat=pu.LpContinuous)
        pi = dict((w,pu.LpVariable('pi_%d' % (w,), cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=prices_raw[w])) for w in winners)
        
        # ebpo objective function
        prob_ebpo += sum(pi.itervalues()) + epsilon*m 
        
        # ebpo constraint: π_j - m <= π_j^vcg 
        for w in winners: prob_ebpo += (pi[w]-m <= prices_vcg[w], 'pi_constr_%d' % w)
        ebpo_solver = SOLVER_CLASS(msg=SOLVER_MSG, mip=False)
        
        # initialize obj_value_sep and prices_t_sum vars, used for comparisons
        obj_value_sep = obj_value_sep_last = 0
        prices_t_sum = prices_t_sum_last = sum(prices_t.itervalues())
        
        # get problem vars
        x, y, _cons = prob_vars
        
        # store revenues_blocking_coalitions with coalitions as key
        revenues_coalitions = []
        
        # store information about the steps (in order to get insights / draw graphs of the process)
        step_info = [{'raw':revenue_raw,'vcg':sum(prices_vcg.itervalues())}]
        
        # abort after 1000 iterations (this should never happen)
        for cnt in xrange(1,1000):
            # make a copy of prob_gwd, since we are using it as basis
            prob_sep = prob_gwd.deepcopy()
            prob_sep.name = 'sep_%d' % cnt
            # build sep t variable
            t = dict((w,pu.LpVariable('t_%d' % (w,), cat=pu.LpBinary)) for w in winners)
            # add the 'maximum coalition contribution' to the objective fn
            prob_sep.objective -= sum((prices_raw[w]-prices_t[w])*t[w] for w in winners)
            
            # add all sep constraints, setting y_i <= t_i
            for w in winners: prob_sep += y[w] <= t[w]
            
            # solve it
            logging.info('sep:\tcalculating - step %d' % cnt)
            solver_status = prob_sep.solve(self.gwd.solver)
            logging.info('sep:\tstatus: %s' % pu.LpStatus[solver_status])
            
            # save the value: z(π^t)
            obj_value_sep, obj_value_sep_last = pu.value(prob_sep.objective), obj_value_sep
            
            # check for a blocking coalition. if no coalition exists, break 
            blocking_coalition_exists = obj_value_sep > prices_t_sum
            if not blocking_coalition_exists:
                logging.info('sep:\tvalue: %d, blocking: None' % obj_value_sep)
                break

            # get the blocking coalition
            blocking_coalition = frozenset(bidder_id for (bidder_id,y_j) in y.iteritems() if round(y_j.value())==1)
            logging.info('sep:\tvalue: %d, blocking: %s' % (obj_value_sep, sorted(blocking_coalition),))
            
            # find out if the currently blocking coalition generates a higher revenue 
            blocking_coalition_with_revenue = dict((b_id,b.budget) for (b_id,b) in bidderInfos.iteritems() if b_id in blocking_coalition)
            revenue_blocking_coalition = sum(blocking_coalition_with_revenue.itervalues())
             
            blocking_coalition_revenue_difference = revenue_blocking_coalition - revenue_raw                                  
            logging.info('sep:\tvalue_blocking_coalition: %d, diff: %s' % (sum(blocking_coalition_with_revenue.itervalues()), blocking_coalition_revenue_difference))

            winners_nonblocking = winners - blocking_coalition
            winners_blocking = winners & blocking_coalition
            
            # update the blocking coalitions value
            revenues_coalitions.append((winners_nonblocking,winners_blocking,revenue_blocking_coalition))
            
            # blocking_y_vals = frozenset(bidder_id for (bidder_id,t_j) in t.iteritems() if round(t_j.value())==1)
            # logging.info('sep:\tblocking y values: %s' % (sorted(blocking_y_vals),))
            
            # SWITCH_COALITION algorithm: switch the winning coalition, if it generates more revenue than the current one
            if blocking_coalition_revenue_difference > 0 and self.algorithm==self.SWITCH_COALITION:
                
                # update all variables related to the gwd
                revenue_raw = revenue_blocking_coalition
                winners = blocking_coalition
                prices_raw = blocking_coalition_with_revenue
                winners_slots = self.gwd.getSlotAssignments(winners, x)
                
                # get the revenues_without_bidders for all new bidders
                prob_vcg = prob_gwd.copy()
                revenues_without_bidders.update(
                    (w, self.vcg.solveStep(prob_vcg,prob_vars,w)) for w in blocking_coalition if w not in revenues_without_bidders
                )
                
                # update prices_vcg using the (new) revenue
                prices_vcg = self.vcg.getPricesForBidders(bidderInfos, revenue_raw, revenues_without_bidders)
                
                # update prices_t
                prices_t = prices_vcg.copy()
                
                # update step_info
                step_info.append({'raw':revenue_raw,'vcg':sum(prices_vcg.itervalues()),'blocking_coalition':revenue_blocking_coalition,'sep':obj_value_sep})
                
                # recreate the ebpo
                pi = dict((w,pu.LpVariable('pi_%d' % (w,), cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=blocking_coalition_with_revenue[w])) for w in winners)
                prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)
                prob_ebpo += sum(pi.itervalues()) + epsilon*m
                
                # ebpo: add vcg price constraints 
                for w in blocking_coalition: prob_ebpo += (pi[w]-m <= prices_vcg[w], 'pi_constr_%d' % w)
                
                # ebpo: add pi constraints
                for nr, (bidders_past_nb,bidders_past_b,revenue_past) in enumerate(revenues_coalitions):
                    winners_past_nb = winners & bidders_past_nb
                    winners_past_b = winners & bidders_past_b
                    if winners_past_nb and winners_past_b:
                        revenue_without_blocking = revenue_past - sum(bidderInfos[wb].budget for wb in winners_past_b)
                        prob_ebpo += (sum(pi[wnb] for wnb in winners_past_nb) >= revenue_without_blocking,'ebpo_constr_%d' % nr)
                    # start a new round for the sep (since we recreated the prices_t vector with the vcg values)
            else:
                winners_nonblocking = winners - blocking_coalition
                winners_blocking = winners & blocking_coalition
                
                logging.info('sep:\twinners: blocking: %s non-blocking: %s' % (sorted(winners_blocking),sorted(winners_nonblocking)))
                
                # add (iteratively) new constraints to the ebpo problem.
                # original formulation:
                # revenue_without_blocking = obj_value_sep - sum(prices_t[wb] for wb in winners_blocking)
                # new formulation:
                revenue_without_blocking = revenue_blocking_coalition - sum(bidderInfos[wb].budget for wb in winners_blocking)
                
                # TRIM_VALUES algorithm: simply trim the revenue_without_blocking (enable the following line)
                # revenue_without_blocking can be at most the sum of the prices_raw of winners_nonblocking
                if self.algorithm==self.TRIM_VALUES:
                    revenue_without_blocking = min(sum(prices_raw[wnb] for wnb in winners_nonblocking), revenue_without_blocking)
                
                # ebpo: add pi constraints
                prob_ebpo += (sum(pi[wnb] for wnb in winners_nonblocking) >= revenue_without_blocking, 'ebpo_constr_%d' % cnt)
                
            # ebpo: solve (this problem can be formulated as a continuous LP).
            prob_ebpo.writeLP('ebpo.lp')
            logging.info('ebpo:\tcalculating - step %s' % (cnt),)
            solver_status = prob_ebpo.solve(ebpo_solver)
            
            # update the π_t list. sum(π_t) has to be equal or increase for each t
            prices_t = dict((int(b.name[3:]), b.varValue) for b in prob_ebpo.variables() if b.name[:2]=='pi')
            prices_t_sum, prices_t_sum_last = round(sum(prices_t.itervalues()),2), prices_t_sum
            
            logging.info('ebpo:\trevenue %d\tprices: %s' % (prices_t_sum,[(k,round(v,2)) for (k,v) in prices_t.iteritems()]))
            assert solver_status == pu.LpStatusOptimal, 'ebpo: not optimal - %s' % pu.LpStatus[solver_status]
            
            assert prices_t_sum >= prices_t_sum_last, 'ebpo: decrease between steps'

#            assert revenue_without_blocking_check == revenue_without_blocking_check_2, ("no: %s %s" % (revenue_without_blocking_check,revenue_without_blocking_check_2))
            # update step_info
            step_info.append({'blocking_coalition':revenue_blocking_coalition,'ebpo':prices_t_sum,'sep':obj_value_sep})
            
            # if both z(π^t) == z(π^t-1) and θ^t == θ^t-1, we are in a local optimum we cannot escape
            # proposition: this happens when the gwd returned a suboptimal solution, causing a winner allocation.
            if obj_value_sep == obj_value_sep_last and prices_t_sum == prices_t_sum_last:
                logging.warn('core:\tvalue did not change. aborting.') 
                break
        else:
            logging.warn('core:\ttoo many iterations in core calculation. aborting.')
        
        # there is no blocking coalition -> the current iteration of the ebpo contains core prices
        revenue_core = sum(prices_t.itervalues())
        logging.info('core:\trevenue %d\tprices: %s' % (revenue_core,[(k,round(v,2)) for (k,v) in prices_t.iteritems()]))
        return (revenue_core, prices_t, prices_raw, prices_vcg, winners_slots, step_info)

class TvAuctionProcessor(object):
    def __init__(self):
        self.gwdClass = Gwd
        self.vcgClass = Vcg
        self.reservePriceClass = ReservePrice
        self.coreClass = CorePricing
        
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
        
        # add a gap and timelimit if set.
        if epgap is not None: gwd.solver.epgap = epgap
        
        if timeLimitGwd is not None: gwd.solver.timeLimit = timeLimitGwd
        elif timeLimit is not None: gwd.solver.timeLimit = timeLimit
        
        _solver_status, (revenue_raw, prices_raw, winners) = gwd.solve(prob_gwd, bidderInfos, prob_vars)
        
        # get the slots for the winners
        x, _y, _cons = prob_vars
        winners_slots = gwd.getSlotAssignments(winners, x)

        # if timelimit was set: adjust it
        if timeLimit is not None: gwd.solver.timeLimit = timeLimit
        
        # solve vcg
        vcg = self.vcgClass(gwd)
        _revenue_vcg, _prices_vcg, revenues_without_bidders = vcg.solve(slots, bidderInfos, revenue_raw, winners, prob_gwd, prob_vars)

        # solve core pricing problem
        core = self.coreClass(gwd, vcg)
        _revenue_core, prices_core, prices_raw, prices_vcg, winners_slots_core, step_info = core.solve(prob_gwd, bidderInfos, winners, prices_raw, revenues_without_bidders, prob_vars)
        if winners_slots_core: winners_slots = winners_slots_core
        
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
