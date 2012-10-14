# -*- coding: utf-8; -*-

import logging
import pulp as pu

SOLVER_MSG = False
SOLVER_CLASS = pu.GUROBI

class Gwd(object):
    '''general winner determination'''
    def __init__(self, slots, bidder_infos):
        self.slots = slots
        self.bidder_infos = bidder_infos
        self.solver = SOLVER_CLASS(msg=SOLVER_MSG)
        
        self.gaps = [] # will contain all gaps used during calculation 
        
    def generate(self):
        '''the winner determination, implemented as a multiple knapsack problem'''
        slots, bidder_infos = self.slots, self.bidder_infos
        
        # x determines whether a bidder can air in a certain slot
        x = pu.LpVariable.dicts('x', (slots.keys(),bidder_infos.keys()), cat=pu.LpBinary)
        # y determines whether a winner has won
        y = pu.LpVariable.dicts('y', (bidder_infos.keys(),), cat=pu.LpBinary)
        # initialize constraints
        cons = []
        # the sum of all assigned ad lengths has to be at most the length of the slot
        for (i,slot) in slots.iteritems():
            f = sum(bidder_info.length*x[i][j] for (j,bidder_info) in bidder_infos.iteritems())
            cons.append( (f <= slot.length , 'slotlen_constr_%d' % i) )
        # match the bidders' demands regarding their attributes
        # attrib_values has to be a list with the same length
        # as the slots list
        for (j,bidder_info) in bidder_infos.iteritems():
            assert len(slots) == len(bidder_info.attrib_values)
            M = sum(bidder_info.attrib_values.itervalues())+1
            f = sum(attrib_value*x[i][j] for (i,attrib_value) in bidder_info.attrib_values.iteritems())
            f2 = bidder_info.attrib_min - f
            cons.append( (f <= M*y[j], 'M_constr_%d.1' % j) )
            cons.append( (f2 <= M*(1-y[j]), 'M_constr_%d.2' % j) )
        # user can at most spend the maximum price
        for (j,bidder_info) in bidder_infos.iteritems():
            f = sum(bidder_info.length*slot.price*x[i][j] for (i,slot) in slots.iteritems())
            cons.append( (f <= bidder_info.budget, 'budget_constr_%d' % j) )
        # oovar domain=bool takes already care of min and max bounds
        prob = pu.LpProblem('gwd', pu.LpMaximize)
        prob += sum(bidder_info.budget*y[j] for (j,bidder_info) in bidder_infos.iteritems())
        # add constraints to problem
        for con in cons: prob += con
        return prob, (x,y,cons)
    
    def solve(self, prob, prob_vars):
        logging.info('wdp:\tcalculating...')
        
        solver_status = prob.solve(self.solver)
        self.gaps.append( (prob.name, prob.solver.epgap_actual) )
        _x, y, _cons = prob_vars
        logging.info('wdp:\tstatus: %s, gap: %.2f' % (pu.LpStatus[solver_status], prob.solver.epgap_actual))
        winners = frozenset(j for (j,y_j) in y.iteritems() if round(y_j.varValue)==1)
        
        revenue_bid = pu.value(prob.objective)
        prices_bid = dict((w,self.bidder_infos[w].budget) for w in winners)
        
        logging.info('bid:\trevenue %d, prices: %s' % (revenue_bid,sorted(prices_bid.iteritems())))
        return solver_status,(revenue_bid, prices_bid, winners)
    
    def getSlotAssignments(self, winners, x):
        winners_slots = dict((w,[]) for w in winners)
        for slot_id, slot_user_vars in x.iteritems():
            slot_winners = [user_id for (user_id,has_won) in slot_user_vars.iteritems() if user_id in winners and has_won.varValue and round(has_won.varValue) == 1]
            for slot_winner in slot_winners: winners_slots[slot_winner].append(slot_id)
        return winners_slots
    
    def getCoalitionValue(self, coalition):
        return sum(self.bidder_infos[j].budget for j in coalition) if coalition else 0

    
class ReservePrice(object):
    def __init__(self, gwd):
        self.gwd = gwd
    '''checks if all winners have to pay at least the reserve price.
    if this is not the case, their price is changed accordingly'''
    def solve(self, winners_slots, prices_before):
        slots, bidder_infos = self.gwd.slots, self.gwd.bidder_infos
        prices_after = {}
        for w,slot_ids_won in winners_slots.iteritems():
            bidder_info = bidder_infos[w]
            price_reserve = sum(slots[slot_id].price for slot_id in slot_ids_won)*bidder_info.length
            prices_after[w] = max(price_reserve, prices_before[w])
        revenue_after = sum(prices_after.itervalues())
        return revenue_after, prices_after

class InitialPricing(object):
    def __init__(self, gwd):
        self.gwd = gwd
        
    def getCoalitions(self, revenue_bid, winners, prob_vcg, prob_vars):
        logging.info('vcg:\tcalculating...')
        winners_without_bidders = {}
        for w in winners:
            winners_without_bidders[w] = self.solveStep(prob_vcg, prob_vars, w)[0]
        return winners_without_bidders
        
    def getPricesForBidders(self, revenue_bid, winners, winners_without_bidders):
        bidder_infos = self.gwd.bidder_infos
        res = {}
        for w in winners:
            revenue_without_bidder = sum(bidder_infos[wo].budget for wo in winners_without_bidders[w])
            res[w] = self.getPriceForBidder(bidder_infos[w].budget, revenue_bid, revenue_without_bidder)
        return res
        
class Zero(InitialPricing):
    '''the zero pricing vector class'''
        
    def solveStep(self, prob_vcg, prob_vars, winner_id):
        return frozenset(), None
    
    def getPriceForBidder(self, budget, revenue_bid, revenue_without_bidder):
        return 0
    
class Vcg(InitialPricing):
    def __init__(self, gwd):
        self.gwd = gwd
    
    def solveStep(self, prob_vcg, prob_vars, winner_id):
        '''takes the original gwd problem, and forces a winner to lose. used in vcg calculation'''
        x, y, _cons = prob_vars
        
        logging.info('vcg:\tcalculating - without winner %s' % (winner_id,))
        prob_vcg.name = 'vcg_%d' % winner_id
        
        # save the original coefficient
        winner_coeff = prob_vcg.objective.get(y[winner_id])
        prob_vcg.objective[y[winner_id]] = 0
        
        solver_status = prob_vcg.resolve(self.gwd.solver)
        self.gwd.gaps.append( (prob_vcg.name, prob_vcg.solver.epgap_actual) )
        logging.info('vcg:\tstatus: %s, gap: %.2f' % (pu.LpStatus[solver_status], prob_vcg.solver.epgap_actual))
        winners_without_w = frozenset(j for (j,y_j) in y.iteritems() if round(y_j.varValue)==1)
        winners_slots = self.gwd.getSlotAssignments(winners_without_w,x)
        
        # restore coefficient
        if winner_coeff is not None: prob_vcg.objective[y[winner_id]] = winner_coeff
        else: del prob_vcg.objective[y[winner_id]]
        return winners_without_w, winners_slots
    
    def getPriceForBidder(self, budget, revenue_bid, revenue_without_bidder):
        return max(0, budget - max(0, (revenue_bid-revenue_without_bidder)))
    
class CorePricing(object):
    
    TRIM_VALUES=1
    SWITCH_COALITIONS=2
    REUSE_COALITIONS=3


    def __init__(self, gwd, vcg, algorithm=REUSE_COALITIONS):
        self.gwd = gwd
        self.vcg = vcg
        self.algorithm = algorithm
    
    
    def _modifySep(self, name, winners, prices_t, prob_sep, y):
        # make a copy of prob_gwd, since we are using it as basis
        prob_sep.name = name
        
        # modify prices according to prices_t. 
        for w,price_t in prices_t.iteritems():
            prob_sep.objective[y[w]] = price_t
            
        # all others get their budgets as prices (this is needed because maybe the winning coalition changed)
        loosers_with_budget = ( (j,bidder_info.budget) for (j,bidder_info) in self.gwd.bidder_infos.iteritems() if j not in winners)
        for j,budget_j in loosers_with_budget:
            prob_sep.objective[y[j]] = budget_j
    
    
    def _createEbpo(self, name, winners, coalitions, prices_vcg):
        bidder_infos = self.gwd.bidder_infos
        # init ebpo
        prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)
        prob_ebpo.name = name
        
        # constants: ε (should be 'small enough')
        epsilon = 1e-64
        
        # variables: m, π_j 
        m = pu.LpVariable('m',cat=pu.LpContinuous)
        pi = dict((w,pu.LpVariable('pi_%d' % (w,), cat=pu.LpContinuous, lowBound=prices_vcg[w], upBound=bidder_infos[w].budget)) for w in winners)
        
        # add objective function
        prob_ebpo += sum(pi.itervalues()) + epsilon*m
        
        # add pi constraints
        for w in winners: prob_ebpo += (pi[w]-m <= prices_vcg[w], 'pi_constr_%d' % w)
        
        # add coalition constraints for all coalitions without the winning coalition
        if self.algorithm==self.REUSE_COALITIONS:
            for c in coalitions-{winners}:
                # \sum_{j \in W \setminus C} \pi_j >= \sum_{j \in C} b_j - \sum_{j \in W \cap C} b_j = \sum_{j \in C \setminus W} b_j 
                prob_ebpo += sum(pi[wnb] for wnb in winners-c) >= sum(bidder_infos[j].budget for j in c-winners)
        return prob_ebpo, pi
    
    
    def _updateToBestCoalition(self, new_coalition, new_coalition_winners_slots, coalitions, winners_without_bidders, prob_vcg, prob_vars):
        '''iteratively update to the best coalition, generating new coalitions while getting vcg coalitions on the way.
        @param new_coalition: the coalition to add to the set.
        @param new_coalition_winners_slots: thew winners_slots dict for new_coalition
        @param coalitions: the coalition set. IN-PLACE change
        @param winners_without_bidders: IN-PLACE change
        '''        
        coalition_changed = False
        winners_slots = None
        best_coalition = self._getBestCoalition(coalitions)
        dirty = False
        isBetter = lambda what,compared_to: self.gwd.getCoalitionValue(what) > self.gwd.getCoalitionValue(compared_to)
        
        if new_coalition:        
            coalitions.add(new_coalition)
        if new_coalition and isBetter(new_coalition,best_coalition):
            best_coalition = new_coalition
            winners_slots = new_coalition_winners_slots
            coalition_changed = True
            dirty = True
        while dirty:
            dirty = False
            missing_winners = (j for j in best_coalition if j not in winners_without_bidders)
            for j in missing_winners:
                coalition_without_j, coalition_slots_without_j = self.vcg.solveStep(prob_vcg,prob_vars,j)
                winners_without_bidders[j] = coalition_without_j 
                coalitions.add(coalition_without_j)
                if isBetter(coalition_without_j,best_coalition):
                    best_coalition = coalition_without_j
                    winners_slots = coalition_slots_without_j
                    coalition_changed = True
                    dirty = True
        return best_coalition if coalition_changed else None, winners_slots
                    
    def _getBestCoalition(self, coalitions):
        return max(coalitions, key=self.gwd.getCoalitionValue) if coalitions else None

    
    def solve(self, prob_gwd, winners, winners_slots, winners_without_bidders, prob_vars):
        
        coalitions = set()
        bidder_infos = self.gwd.bidder_infos
        winners_without_bidders = winners_without_bidders.copy()
        
        # we need a prob_vcg because of the generation of vcg prices for new coalition entries
        prob_vcg = prob_gwd
        prob_sep = prob_gwd
        
        # init ebpo and variables
        prob_ebpo = None
        ebpo_solver = SOLVER_CLASS(msg=SOLVER_MSG, mip=False)
        pi = {}
        
        
        # get currently highest valued coalition
        if self.algorithm in (self.REUSE_COALITIONS, self.SWITCH_COALITIONS):
            winners, winners_slots = self._updateToBestCoalition(winners, winners_slots, coalitions, winners_without_bidders, prob_vcg, prob_vars)
        revenue_bid = self.gwd.getCoalitionValue(winners)
        
        # init prices_vcg and prices_t
        prices_vcg = self.vcg.getPricesForBidders(revenue_bid, winners, winners_without_bidders)
        prices_t = prices_vcg.copy()
        
        # initialize obj_value_sep and prices_t_sum vars, used for comparisons
        obj_value_sep = obj_value_sep_last = 0
        prices_t_sum = prices_t_sum_last = sum(prices_t.itervalues())
        blocking_coalition = blocking_coalition_last = None
        
        # store information about the steps (in order to get insights / draw graphs of the process)
        step_info = [{'bid':revenue_bid,'vcg':sum(prices_vcg.itervalues())}]
        
        # get problem vars
        x, y, _cons = prob_vars
        
        # abort after 1000 iterations (this should never happen)
        for cnt in xrange(1000):
            
            # create sep
            self._modifySep('sep_%d' % cnt, winners, prices_t, prob_sep, y)
            
            # solve it
            logging.info('sep:\tcalculating - step %d' % cnt)
            solver_status = prob_sep.resolve(self.gwd.solver)
            self.gwd.gaps.append( (prob_sep.name, prob_sep.solver.epgap_actual) )
            logging.info('sep:\tstatus: %s, gap: %.2f' % (pu.LpStatus[solver_status], prob_sep.solver.epgap_actual))
            
            # save the value: z(π^t)
            obj_value_sep, obj_value_sep_last = pu.value(prob_sep.objective), obj_value_sep
            
            # check for no blocking coalition exists. if yes, break
            if not obj_value_sep > sum(prices_t.itervalues()):
                logging.info('sep:\tvalue: %d, blocking: None' % obj_value_sep)
                break
            
            # get the blocking coalition
            blocking_coalition, blocking_coalition_last = frozenset(bidder_id for (bidder_id,y_j) in y.iteritems() if round(y_j.varValue)==1), blocking_coalition
            blocking_coalition_slots = self.gwd.getSlotAssignments(blocking_coalition, x)
            logging.info('sep:\tvalue: %d, coalition: value: %d, members: %s' % (obj_value_sep, self.gwd.getCoalitionValue(blocking_coalition), sorted(blocking_coalition)))
            
            # check if the blocking coalition is better. if yes update the winning coalition
            if self.algorithm in (self.SWITCH_COALITIONS,self.REUSE_COALITIONS):
                new_winners, new_winners_slots = self._updateToBestCoalition(blocking_coalition, blocking_coalition_slots, coalitions, winners_without_bidders, prob_vcg, prob_vars)
                if new_winners: winners, winners_slots = new_winners, new_winners_slots
                
            # if there is no ebpo or if the winners changed and we use an appropriate algorithm, recreate the ebpo
            if prob_ebpo is None or self.algorithm in (self.SWITCH_COALITIONS, self.REUSE_COALITIONS) and new_winners:
                revenue_bid = self.gwd.getCoalitionValue(winners)
                prices_vcg = self.vcg.getPricesForBidders(revenue_bid, winners, winners_without_bidders)
                logging.info('ebpo:\tcreating - coalition value: %d, winners: %s' % (self.gwd.getCoalitionValue(winners),sorted(winners)))
                prob_ebpo, pi = self._createEbpo('ebpo_%d' % cnt, winners, coalitions, prices_vcg)
                
            # else if the winners did not change and we are using switch/reuse, add the constraint
            elif self.algorithm in (self.SWITCH_COALITIONS, self.REUSE_COALITIONS) and not new_winners:
                prob_ebpo += sum(pi[wnb] for wnb in winners-blocking_coalition) >= sum(bidder_infos[j].budget for j in blocking_coalition-winners)
            
            # else if we trim the values, we always call this block
            elif self.algorithm==self.TRIM_VALUES:
                prob_ebpo += sum(pi[wnb] for wnb in winners-blocking_coalition) >= min(
                    sum(bidder_infos[j].budget for j in blocking_coalition-winners), # original right-hand side of constraint 
                    sum(bidder_infos[j].budget for j in winners-blocking_coalition)  # can be at most b_j forall j in W\C^t
                )
            else: raise Exception('logical error')
                
                
            # ebpo: solve (this problem can be formulated as a continuous LP).
            logging.info('ebpo:\tcalculating - step %s' % cnt)
            solver_status = prob_ebpo.solve(ebpo_solver)
            
            # update the π_t list. sum(π_t) has to be equal or increase for each t (except when changing winners)
            prices_t = dict((j,pi_j.varValue or pi_j.lowBound) for (j,pi_j) in pi.iteritems())
            prices_t_sum, prices_t_sum_last = round(sum(prices_t.itervalues()),2), prices_t_sum
            # ebpo: has to be feasible
            assert solver_status == pu.LpStatusOptimal, 'ebpo: not optimal - %s' % pu.LpStatus[solver_status]
            logging.info('ebpo:\tvalue: %d' % prices_t_sum)
            
            # update step_info
            step_info.append({'ebpo':prices_t_sum,'sep':obj_value_sep,'bid':revenue_bid,'vcg':sum(prices_vcg.itervalues())})
            
            # if both z(π^t) == z(π^t-1) and θ^t == θ^t-1, we are in a local optimum we cannot escape
            # proposition: this happens when the gwd returned a suboptimal solution, causing a winner allocation.
            if obj_value_sep == obj_value_sep_last and prices_t_sum == prices_t_sum_last and blocking_coalition == blocking_coalition_last:
                logging.warn('core:\tvalue did not change. aborting.') 
                break
        else:
            logging.warn('core:\ttoo many iterations in core calculation. aborting.')
            
        revenue_core = sum(prices_t.itervalues())
        logging.info('core:\trevenue %d, prices: %s' % ( revenue_core, sorted((j,round(pi_j,2)) for (j,pi_j) in prices_t.iteritems()) ) )
        return winners_slots, prices_t, prices_vcg, step_info
        
class TvAuctionProcessor(object):
    def __init__(self):
        self.gwdClass = Gwd
        self.vcgClass = Vcg
        self.reservePriceClass = ReservePrice
        self.coreClass = CorePricing
        self.core_algorithm = CorePricing.REUSE_COALITIONS
        
    def isOptimal(self,solver_status):
        return solver_status == pu.LpStatusOptimal
    
    def solve(self, slots, bidder_infos, timeLimit=20, timeLimitGwd=20, epgap=None):
        '''solve the wdp and pricing problem.
        
        @param slots:        a dict of Slot objects
        @param bidder_infos:  a dict of BidderInfo objects
        @param timeLimit:    int|null, sets the time limit for all integer problems.
        @param timeLimitGwd: int|null, lets you override the time limit for the gwd because of its importance. 
        @param epgap:        float|null, is used for all integer problems.'''
        
        # generate the gwd    
        gwd = self.gwdClass(slots, bidder_infos)
        prob_gwd, prob_vars = gwd.generate()
        x, _y, _cons = prob_vars
        
        gwd.solver.epgap = epgap
        gwd.solver.timeLimit = timeLimitGwd
        
        # solve gwd
        _solver_status, (revenue_bid, _prices_bid, winners) = gwd.solve(prob_gwd, prob_vars)
        
        # get the slots for the winners (usually re-set if we switch/reuse coalitions) 
        winners_slots = gwd.getSlotAssignments(winners, x)
        # re-set timeLimit
        gwd.solver.timeLimit = timeLimit
        
        # solve vcg (only if we trim)
        # the vcg prices/coalitions are generated iteratively in the core class when we switch/reuse coalitions
        vcg = self.vcgClass(gwd)
        winners_without_bidders = {}
        if self.core_algorithm == CorePricing.TRIM_VALUES:
            prob_vcg = prob_gwd
            winners_without_bidders = vcg.getCoalitions(revenue_bid, winners, prob_vcg, prob_vars)

        # solve core pricing problem
        core = self.coreClass(gwd, vcg, self.core_algorithm)
        winners_slots_core, prices_core, prices_vcg, step_info = core.solve(prob_gwd, winners, winners_slots, winners_without_bidders, prob_vars)
        
        # if the winners set was recreated
        if winners_slots_core: winners_slots = winners_slots_core
        winners = frozenset(winners_slots)
        
        # raise prices to the reserve price if needed
        reservePrice = self.reservePriceClass(gwd)
        _revenue_after, prices_after = reservePrice.solve(winners_slots, prices_core)
        
        
        return {
            'winners': sorted(winners),
            'winners_slots': winners_slots,
            'prices_bid': dict((w,bidder_infos[w].budget) for w in winners),
            'prices_vcg': prices_vcg,
            'prices_core': prices_core,
            'prices_final': prices_after,
            'step_info': step_info,
            'gaps': gwd.gaps
        }
     
if __name__ == '__main__':
    import os
    import sys
    from optparse import OptionParser
    from common import json, convertToNamedTuples

    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    logging.basicConfig(level=log_level)
    
    parser = OptionParser()
    parser.set_usage('%prog [options] < scenarios.json')
    parser.add_option('--initial-vector', dest='price_vector', choices=('vcg','zero'), default='vcg', help='the type of price vector used as a starting point for core price generation (vcg,zero)')
    parser.add_option('--core-algorithm', dest='core_algorithm', choices=('trim','switch','reuse'), default='reuse', help='which algorithm should be used in case a suboptimal winner determination is discovered during core pricing (trim: trim the values to be within a feasible region, switch: recreate the ebpo,reuse: recreate the ebpo and try to re-use already existing constraints)')
    parser.add_option('--time-limit-gwd',dest='time_limit_gwd', type='int', default='20', help='the time limit for the initial winner determination problem')
    parser.add_option('--time-limit',dest='time_limit', type='int', default='20', help='the time limit for all problems but the initial winner determination problem')
    parser.add_option('--epgap',dest='epgap', type='float', default=None, help='the epgap used for all problems')
    parser.add_option('--offset',dest='offset', type='int', default=0, help='the scenario offset used')
    for option in parser.option_list: 
        if option.default != ("NO", "DEFAULT"): option.help += (" " if option.help else "") + "[default: %default]"
    if sys.stdin.isatty():
        print parser.format_help()
        sys.exit()
    
    # parse scenario and options    
    scenarios = json.decode(sys.stdin.read())
    options = parser.parse_args()[0]

    scenario = scenarios[options.offset]
    convertToNamedTuples(scenario)
    slots, bidder_infos = scenario
    
    # create processor object and set values
    proc = TvAuctionProcessor()
    if options.price_vector=='vcg': proc.vcgClass = Vcg
    elif options.price_vector=='zero': proc.vcgClass = Zero
    
    if options.core_algorithm=='trim': proc.core_algorithm = CorePricing.TRIM_VALUES
    elif options.core_algorithm=='switch': proc.core_algorithm = CorePricing.SWITCH_COALITIONS
    elif options.core_algorithm=='reuse': proc.core_algorithm = CorePricing.REUSE_COALITIONS
    
    # solve and print
    res = proc.solve(slots, bidder_infos, options.time_limit, options.time_limit_gwd, options.epgap)
    print json.encode(res)
