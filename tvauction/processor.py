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
        self.coalitions = {} # contains coalitions and their value
        self.gaps = [] # will contain all gaps used during calculation
        
    
    def generate(self):
        '''the winner determination, implemented as a multiple knapsack problem'''
        slots, bidder_infos = self.slots, self.bidder_infos
        
        # x determines whether a bidder can air in a certain slot
        x = {}
        for i in slots:
            x[i] = {}
            for (k,bidder_info) in bidder_infos.iteritems():
                x[i][k] = dict((j,pu.LpVariable('x_%d_%d_%d' % (i,k,j), cat=pu.LpBinary)) for j in range(len(bidder_info.bids)))
                
        # y determines whether a bid has won
        y = {}
        for (k,bidder_info) in bidder_infos.iteritems():
            y[k] = dict((j,pu.LpVariable('y_%d_%d' % (k,j), cat=pu.LpBinary)) for j in range(len(bidder_info.bids)))
            
        # initialize constraints
        cons = []
        
        # the sum of all assigned ad lengths has to be at most the length of the slot
        for (i,slot) in slots.iteritems():
            f = sum(bidder_info.length*sum(x[i][k].itervalues()) for (k,bidder_info) in bidder_infos.iteritems())
            cons.append( (f <= slot.length , 'slotlen_constr_%d' % i) )
            
        # match the bidders' demands regarding their attributes.
        # attrib_values has to be a list with the same length as the slots list
        for (k,bidder_info) in bidder_infos.iteritems():
            assert len(slots) == len(bidder_info.attrib_values)
            M = sum(bidder_info.attrib_values.itervalues())*2
            for (j, (_bid_price, attrib_min)) in enumerate(bidder_info.bids):
                f = sum(attrib_value*x[i][k][j] for (i,attrib_value) in bidder_info.attrib_values.iteritems())
                f2 = attrib_min - f
                cons.append( (f <= M*y[k][j], 'M_constr_%d_%d.1' % (k,j)) )
                cons.append( (f2 <= M*(1-y[k][j]), 'M_constr_%d_%d.2' % (k,j)) )
                
        # user can at most spend the maximum price
        for (k,bidder_info) in bidder_infos.iteritems():
            for (j, (bid_price, _attrib_min)) in enumerate(bidder_info.bids):
                f = sum(bidder_info.length*slot.price*x[i][k][j] for (i,slot) in slots.iteritems())
                cons.append( (f <= bid_price, 'budget_constr_%d_%d' % (k,j)) )
                
        # bidder can win at most 1 time (xor)
        for (k, bidder_info) in bidder_infos.iteritems():
            f = sum(y[k][j] for j in range(len(bidder_info.bids)))
            cons.append( (f <= 1) )
                
        # oovar domain=bool takes already care of min and max bounds
        prob = pu.LpProblem('gwd', pu.LpMaximize)
        prob += sum(sum(budget*y[k][j] for (j, (budget,_attrib_min)) in enumerate(bidder_info.bids)) for (k,bidder_info) in bidder_infos.iteritems())
        # add constraints to problem
        for con in cons: prob += con
        return prob, (x,y,cons)
    
    
    def solve(self, prob, prob_vars):
        x, y, _cons = prob_vars
        logging.info('wdp:\tcalculating...')
        
        solver_status = prob.solve(self.solver)
#        self.printMatrix(x,y)
        winners = self.getWinningCoalition(y)
        winners_slots = self.getSlotAssignments(winners, x)
        self.addToCoalitions(winners, self.calculateCoalitionValue(winners), 'gwd', True)
        self.gaps.append( (prob.name, prob.solver.epgap_actual) )
        
        logging.info('wdp:\tstatus: %s, gap: %.2f' % (pu.LpStatus[solver_status], prob.solver.epgap_actual or -1))
        logging.info('bid:\tvalue %d, members: %s' % (self.getCoalitionValue(winners), sorted(k for (k,_j) in winners)))
        return winners, winners_slots
    
    
    def getWinningCoalition(self, y):
        '''generates a coalition based on the values of the objective variable y. 
        a coalition is a frozen set containing tuples (k,j)'''
        coalition = []
        for k, y_k in y.iteritems():
            for j, y_kj in y_k.iteritems():
                if round(y_kj.value()) == 1:
                    coalition.append((k,j))
        return frozenset(coalition)
    
    def getSlotAssignments(self, coalition, x):
        coalition_slots = dict((k,[]) for (k,j) in coalition)
#        print 'coa', coalition
        for i, x_i in x.iteritems():
            slot_winners = []
            for (k, x_ik) in x_i.iteritems():
                for (j, x_ikj) in x_ik.iteritems():
                    if round(x_ikj.value() or 0) == 1 and (k,j) in coalition:
                        slot_winners.append(k)
#            print slot_winners
            for k in slot_winners: coalition_slots[k].append(i)
        return coalition_slots
    
    def calculateCoalitionValue(self, coalition):
        return sum(self.bidder_infos[k].bids[j][0] for (k,j) in coalition)
        
    def calculateSumPrices(self, prices):
        return sum(prices.itervalues())
        
    def getCoalitionValue(self, coalition):
        return self.coalitions[coalition] if coalition else 0
    
    def addToCoalitions(self, coalition, value, where, is_new_best=None):
        # check if is new best
        if is_new_best is None:
            is_new_best = value > max(self.coalitions.itervalues())
        # add to coalitions is not in there yet    
        if coalition not in self.coalitions:
            self.coalitions[coalition] = value
        # check if coalition is a subset and has a greater value. if yes, update also the others
        for (other_coalition,other_coalition_val) in self.coalitions.iteritems():
            if coalition <= other_coalition and value > other_coalition_val:
                self.coalitions[other_coalition] = value
        if is_new_best:
            logging.info('%s:\tnew best coalition added' % (where,))
        return is_new_best
    
    def printMatrix(self, x, y):
        for k,y_k in y.iteritems():
            for j, y_kj in y_k.iteritems():
                print >>sys.stderr, k,'\t',j,'%d' % y_kj.value(),'\t',
                print >>sys.stderr, ' '.join('%d' % round(x_i[k][j].value()) if x_i[k][j].value() else ' ' for x_i in x.itervalues())
                 
class ReservePrice(object):
    '''checks if all winners have to pay at least the reserve price.
    if this is not the case, their price is changed accordingly'''
    
    def __init__(self, gwd):
        self.gwd = gwd
        
    def solve(self, winners, winners_slots, prices_before):
        slots, bidder_infos = self.gwd.slots, self.gwd.bidder_infos
        prices_after = {}
        for k,slot_ids_won in winners_slots.iteritems():
            bidder_info = bidder_infos[k]
            price_reserve = bidder_info.length*sum(slots[i].price for i in slot_ids_won)
            prices_after[k] = max(price_reserve, prices_before[k])
        revenue_after = self.gwd.calculateSumPrices(prices_after)
        return revenue_after, prices_after

class InitialPricing(object):
    def __init__(self, gwd):
        self.gwd = gwd
        
    def getCoalitions(self, winners, prob_vcg, prob_vars):
        logging.info('vcg:\tcalculating...')
        winners_without_bidders = {}
        for (k,_j) in winners:
            winners_without_bidders[k] = self.solveStep(prob_vcg, prob_vars, k)[0]
        return winners_without_bidders
        
    def getPricesForBidders(self, winners, winners_without_bidders):
        res = {}
        revenue_bid = self.gwd.getCoalitionValue(winners) 
        for (k,j) in winners:
            winner_bid = self.gwd.bidder_infos[k].bids[j][0]
            revenue_without_bidder = self.gwd.getCoalitionValue(winners_without_bidders[k])
            res[k] = self.getPriceForBidder(winner_bid, revenue_bid, revenue_without_bidder) 
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
        
        # save the original coefficients and set them to 0
        winner_coeffs = {}
        for j,y_kj in y[winner_id].iteritems():
            winner_coeffs[j] = prob_vcg.objective.get(y_kj)
            prob_vcg.objective[y_kj] = 0
            
        # solve
        solver_status = prob_vcg.resolve(self.gwd.solver)
#        self.gwd.printMatrix(x,y)
        winners_without_w = self.gwd.getWinningCoalition(y) # - {winner_id} # FIXME
        winners_without_w_slots = self.gwd.getSlotAssignments(winners_without_w, x)
        self.gwd.addToCoalitions(winners_without_w, self.gwd.calculateCoalitionValue(winners_without_w), 'vcg')
        self.gwd.gaps.append( (prob_vcg.name, prob_vcg.solver.epgap_actual) )
        logging.info('vcg:\tstatus: %s, gap: %.2f' % (pu.LpStatus[solver_status], prob_vcg.solver.epgap_actual or -1))
        logging.info('vcg:\tvalue: %d, members: %s' % (self.gwd.getCoalitionValue(winners_without_w), sorted(k for (k,_j) in winners_without_w)))
        
        # restore coefficients
        for j, winner_coeff in winner_coeffs.iteritems():
            if winner_coeff is not None: prob_vcg.objective[y[winner_id][j]] = winner_coeff
            else: del prob_vcg.objective[y[winner_id][j]]
        return winners_without_w, winners_without_w_slots
    
    
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
    
    
    def _modifySep(self, name, winners, objective, prices_t, prob_sep, y):
        # restore the objective from the original copy
        prob_sep.name = name
        prob_sep.objective = objective.copy()
        
        # modify prices according to prices_t
        # we subtract the b_k^* - p_k from each bid, so additional constraints are not needed
        for (k, j_best) in winners:
            b_k_best = self.gwd.bidder_infos[k].bids[j_best][0]
            price_t_k = prices_t[k]
            for j, y_kj in y[k].iteritems():
                bid_jk = self.gwd.bidder_infos[k].bids[j][0]
                prob_sep.objective[y_kj] = bid_jk - (b_k_best + price_t_k)
    
    def _createEbpo(self, name, winners, prices_vcg):
        bidder_infos = self.gwd.bidder_infos
        # init ebpo
        prob_ebpo = pu.LpProblem('ebpo',pu.LpMinimize)
        prob_ebpo.name = name
        
        # constants: ε (should be 'small enough')
        epsilon = 1.0 / max(bidder_infos[k].bids[j][0] for (k,j) in winners)
        
        # variables: m, π_j 
        m = pu.LpVariable('m',cat=pu.LpContinuous)
        pi = dict((k,pu.LpVariable('pi_%d' % k, cat=pu.LpContinuous, lowBound=prices_vcg[k], upBound=bidder_infos[k].bids[j][0])) for (k,j) in winners)
        # add objective function
        prob_ebpo += sum(pi.itervalues()) + epsilon*m
        
        # add pi constraints
        for (k,j) in winners: prob_ebpo += (pi[k]-m <= prices_vcg[k], 'pi_constr_%d' % k)
        
        # add coalition constraints for all coalitions without the winning coalition
        if self.algorithm==self.REUSE_COALITIONS:
            for c in set(self.gwd.coalitions) - set(winners):
                prob_ebpo += sum(pi[k] for (k,j) in winners-c) >= self.gwd.getCoalitionValue(c) - sum(bidder_infos[k].bids[j][0] for (k,j) in winners&c)
        return prob_ebpo, pi
    
    
    def _updateToBestCoalition(self, where, new_coalition, new_coalition_slots, winners_without_bidders, prob_vcg, prob_vars, is_new_best=None):
        '''iteratively update to the best coalition, generating new coalitions while getting vcg coalitions on the way.
        @param new_coalition: the coalition to add to the set.
        @param new_coalition_best_coalition_slots: the best_coalition_slots dict for new_coalition
        @param coalitions: the coalition set. IN-PLACE change
        @param winners_without_bidders: IN-PLACE change
        '''        
        coalition_changed = False
        best_coalition = self._getBestCoalition(self.gwd.coalitions.keys())
        best_coalition_slots = None
        
        dirty = False
        isBetter = lambda what,compared_to: self.gwd.getCoalitionValue(what) > self.gwd.getCoalitionValue(compared_to)
        
        if is_new_best is None:
            is_new_best = isBetter(new_coalition, best_coalition)
        if is_new_best:
            best_coalition = new_coalition
            best_coalition_slots = new_coalition_slots
            logging.info('%s:\tcoalition changed' % (where,))
            coalition_changed = True
            dirty = True
        while dirty:
            dirty = False
            missing_winners = (k for (k,j) in best_coalition if k not in winners_without_bidders)
            for k in missing_winners:
                coalition_without_k, coalition_slots_without_k = self.vcg.solveStep(prob_vcg,prob_vars,k)
                winners_without_bidders[k] = coalition_without_k 
                coalition_without_k_is_better = isBetter(coalition_without_k, best_coalition)
                if coalition_without_k_is_better:
                    best_coalition = coalition_without_k
                    best_coalition_slots = coalition_slots_without_k
                    logging.info('%s:\tcoalition changed' % (where,))
                    coalition_changed = True
                    dirty = True
                    break
        return best_coalition if coalition_changed else None, best_coalition_slots
                    
    def _getBestCoalition(self, coalitions):
        return max(coalitions, key=self.gwd.getCoalitionValue) if coalitions else None

    def solve(self, prob_gwd, winners, winners_slots, winners_without_bidders, prob_vars):
        
        bidder_infos = self.gwd.bidder_infos
        winners_without_bidders = winners_without_bidders.copy()
        
        # set prob_vcg and prob_sep
        prob_vcg = prob_gwd
        prob_sep = prob_gwd
        
        # we need to copy the objective to be able to restore it on each sep run
        objective = prob_gwd.objective.copy()
        
        # init ebpo and variables
        prob_ebpo = None
        ebpo_solver = SOLVER_CLASS(msg=SOLVER_MSG, mip=False)
        pi = {}
        
        # get currently highest valued coalition
        if self.algorithm in (self.REUSE_COALITIONS, self.SWITCH_COALITIONS):
            winners, winners_slots = self._updateToBestCoalition('sep', winners, winners_slots, winners_without_bidders, prob_vcg, prob_vars, is_new_best=True)
        
        # init prices_vcg and prices_t
        prices_vcg = self.vcg.getPricesForBidders(winners, winners_without_bidders)
        prices_t = prices_vcg.copy()
        
        # initialize obj_value_sep and prices_t_sum vars, used for comparisons
        obj_value_sep = obj_value_sep_last = 0
        prices_t_sum = prices_t_sum_last = self.gwd.calculateSumPrices(prices_t)
        blocking_coalition = blocking_coalition_last = None
        
        # store information about the steps (in order to get insights / draw graphs of the process)
        step_info = [{'bid':self.gwd.getCoalitionValue(winners),'vcg': self.gwd.calculateSumPrices(prices_vcg)}]
        
        # get problem vars
        x, y, _cons = prob_vars
        
        not_changed_cnt = 0
        core_reached_cnt = 0
        # abort after 1000 iterations (this should never happen)
        for cnt in xrange(1000):
            
            # restore objective and solve sep
            logging.info('sep:\tcalculating - step %d' % cnt)
            
            self._modifySep('sep_%d' % cnt, winners, objective, prices_t, prob_sep, y)
            solver_status = prob_sep.resolve(self.gwd.solver)
#            self.gwd.printMatrix(x,y)
            blocking_coalition, blocking_coalition_last = self.gwd.getWinningCoalition(y), blocking_coalition
            blocking_coalition_slots = self.gwd.getSlotAssignments(blocking_coalition, x)
            self.gwd.addToCoalitions(blocking_coalition, self.gwd.calculateCoalitionValue(blocking_coalition), 'sep_%d' % cnt)
            self.gwd.gaps.append( (prob_sep.name, prob_sep.solver.epgap_actual) )
            logging.info('sep:\tstatus: %s, gap: %.2f' % (pu.LpStatus[solver_status], prob_sep.solver.epgap_actual or -1))
            logging.info('sep:\tvalue: %d, coalition value: %d, members: %s' % (obj_value_sep, self.gwd.getCoalitionValue(blocking_coalition), sorted(k for (k,_j) in blocking_coalition)))
            
            # save the value: z(π^t)
            obj_value_sep, obj_value_sep_last = prob_sep.objective.value(), obj_value_sep
            

            # check for no blocking coalition exists. if this happened several times, break
            blocking_coalition_revenue_higher = obj_value_sep > self.gwd.calculateSumPrices(prices_t)
            if not blocking_coalition_revenue_higher and core_reached_cnt >= 2:
                logging.info('sep:\tnot blocking (breaking)')
                step_info.append({'ebpo':prices_t_sum,'sep':obj_value_sep,'bid':self.gwd.getCoalitionValue(winners),'vcg':self.gwd.calculateSumPrices(prices_vcg)})
                break
            elif not blocking_coalition_revenue_higher:
                core_reached_cnt += 1
                logging.info('sep:\tnot blocking (going on)')
                step_info.append({'ebpo':prices_t_sum,'sep':obj_value_sep,'bid':self.gwd.getCoalitionValue(winners),'vcg':self.gwd.calculateSumPrices(prices_vcg)})
                continue
            else:
                core_reached_cnt = 0
            
            # check if the blocking coalition is better. if yes update the winning coalition
            if self.algorithm in (self.SWITCH_COALITIONS,self.REUSE_COALITIONS):
                new_winners, new_winners_slots = self._updateToBestCoalition('sep', blocking_coalition, blocking_coalition_slots, winners_without_bidders, prob_vcg, prob_vars)
                if new_winners: winners, winners_slots = new_winners, new_winners_slots
                
            # if there is no ebpo or if the winners changed and we use an appropriate algorithm, recreate the ebpo
            if prob_ebpo is None or self.algorithm in (self.SWITCH_COALITIONS, self.REUSE_COALITIONS) and new_winners:
                prices_vcg = self.vcg.getPricesForBidders(winners, winners_without_bidders)
                logging.info('ebpo:\tcreating - coalition value: %d, members: %s' % (self.gwd.getCoalitionValue(winners),sorted(k for (k,_j) in winners)))
                prob_ebpo, pi = self._createEbpo('ebpo_%d' % cnt, winners, prices_vcg)
                
            # else if the winners did not change and we are using switch/reuse, add the constraint
            elif self.algorithm in (self.SWITCH_COALITIONS, self.REUSE_COALITIONS) and not new_winners:
                prob_ebpo += sum(pi[k] for (k,j) in winners-blocking_coalition) >= self.gwd.getCoalitionValue(blocking_coalition) - sum(bidder_infos[k].bids[j][0] for (k,j) in winners&blocking_coalition)
            
            # else if we trim the values, we always call this block
            elif self.algorithm==self.TRIM_VALUES:
                prob_ebpo += sum(pi[wnb] for wnb in winners-blocking_coalition) >= min(
                    sum(bidder_infos[k].bids[j][0] for (k,j) in winners-blocking_coalition),
                    self.gwd.getCoalitionValue(blocking_coalition) - sum(bidder_infos[k].bids[j][0] for (k,j) in winners&blocking_coalition)
                )
            else: raise Exception('logical error')
                
            # ebpo: solve (this problem can be formulated as a continuous LP).
            logging.info('ebpo:\tcalculating - step %s' % cnt)
            solver_status = prob_ebpo.solve(ebpo_solver)
            
            # update the π_t list. sum(π_t) has to be equal or increase for each t (except when changing winners)
            prices_t = dict((kj, pi_k.value()) for (kj, pi_k) in pi.iteritems())
            prices_t_sum, prices_t_sum_last = self.gwd.calculateSumPrices(prices_t), prices_t_sum
            # ebpo: has to be feasible
            assert solver_status == pu.LpStatusOptimal, 'ebpo: not optimal - %s' % pu.LpStatus[solver_status]
            logging.info('ebpo:\tvalue: %d' % prices_t_sum)
            
            # update step_info
            step_info.append({'ebpo':prices_t_sum,'sep':obj_value_sep,'bid':self.gwd.getCoalitionValue(winners),'vcg':self.gwd.calculateSumPrices(prices_vcg)})
            
            # if both z(π^t) == z(π^t-1) and θ^t == θ^t-1, we are in a local optimum we cannot escape
            # proposition: this happens when the gwd returned a suboptimal solution, causing a winner allocation.
            coalition_and_values_identical = obj_value_sep == obj_value_sep_last and prices_t_sum == prices_t_sum_last and blocking_coalition == blocking_coalition_last 
            if coalition_and_values_identical: not_changed_cnt += 1
            else: not_changed_cnt = 0
            if not_changed_cnt >= 5:
                logging.warn('core:\tvalue did not change. aborting.') 
                break
        else:
            logging.warn('core:\ttoo many iterations in core calculation. aborting.')
            
        revenue_core = self.gwd.calculateSumPrices(prices_t)
        logging.info('core:\trevenue %d, prices: %s' % ( revenue_core, prices_t))
        return winners, winners_slots, prices_t, prices_vcg, step_info
        
class TvAuctionProcessor(object):
    def __init__(self):
        self.gwdClass = Gwd
        self.vcgClass = Vcg
        self.reservePriceClass = ReservePrice
        self.coreClass = CorePricing
        self.core_algorithm = CorePricing.REUSE_COALITIONS
        
    def solve(self, slots, bidder_infos, timeLimit=20, timeLimitGwd=20, epgap=None):
        '''solve the wdp and pricing problem.
        
        @param slots:        a dict of Slot objects
        @param bidder_infos: a dict of BidderInfo objects
        @param timeLimit:    int|null, sets the time limit for all integer problems.
        @param timeLimitGwd: int|null, lets you override the time limit for the gwd because of its importance. 
        @param epgap:        float|null, is used for all integer problems.'''
        
        # generate the gwd    
        gwd = self.gwdClass(slots, bidder_infos)
        prob_gwd, prob_vars = gwd.generate()
        
        gwd.solver.epgap = epgap
        gwd.solver.timeLimit = timeLimitGwd
        
        # solve gwd
        # get the slots for the winners (usually re-set if we switch/reuse coalitions) 
        winners, winners_slots = gwd.solve(prob_gwd, prob_vars)
        # re-set timeLimit
        gwd.solver.timeLimit = timeLimit
        
        # solve vcg (only if we trim)
        # the vcg prices/coalitions are generated iteratively in the core class when we switch/reuse coalitions
        vcg = self.vcgClass(gwd)
        winners_without_bidders = {}
        if self.core_algorithm == CorePricing.TRIM_VALUES:
            prob_vcg = prob_gwd
            winners_without_bidders = vcg.getCoalitions(winners, prob_vcg, prob_vars)

        # solve core pricing problem
        core = self.coreClass(gwd, vcg, self.core_algorithm)
        winners_core, winners_slots_core, prices_core, prices_vcg, step_info = core.solve(prob_gwd, winners, winners_slots, winners_without_bidders, prob_vars)
        winners, winners_slots = winners_core, winners_slots_core
        
        # raise prices to the reserve price if needed
        reservePrice = self.reservePriceClass(gwd)
        _revenue_after, prices_after = reservePrice.solve(winners, winners_slots, prices_core)
        
        return {
            'winners': sorted(winners),
            'winners_slots': winners_slots,
            'prices_bid': dict((k, bidder_infos[k].bids[j][0]) for (k,j) in winners),
            'prices_vcg': prices_vcg,
            'prices_core': prices_core,
            'prices_final': prices_after,
            'step_info': step_info,
            'gaps': gwd.gaps,
            'coalitions': sorted(gwd.coalitions.iteritems(), key=lambda (k,v): v, reverse=True)
        }
     
if __name__ == '__main__':
    import os
    import sys
    from optparse import OptionParser
    from common import json, convertToNamedTuples

    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    SOLVER_MSG = log_level < logging.INFO 
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
    res = proc.solve(slots, bidder_infos, options.time_limit or None, options.time_limit_gwd or None, options.epgap or None)
    print json.encode(res)
