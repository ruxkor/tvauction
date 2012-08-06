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
    def __init__(self):
        self.solver = pu.GUROBI(msg=False)
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
    
    def solve(self, prob, bidderInfos):
        logging.info('wdp:\tcalculating...')
        
        solver_status = prob.solve(self.solver)
        winners = frozenset(int(v.name[2:]) for v in prob.variables() if v.varValue==1 and v.name[:1]=='y')
        
        revenue_raw = pu.value(prob.objective)
        prices_raw = dict((w,bidderInfos[w].budget) for w in winners)
        
        logging.info('raw:\trevenue %d\tprices: %s' % (revenue_raw,sorted(prices_raw.iteritems())))
        return (solver_status,(revenue_raw, prices_raw, winners))
    
        
class Vcg(object):
    def __init__(self, gwd):
        self.gwd = gwd
        pass
    def calculate(self, slots, bidderInfos, revenue_raw, winners, bidders_satisfied):
        logging.info('vcg:\tcalculating...')
        prices_vcg = {}
        for w in winners:
            bidderinfo_winner = bidderInfos[w]
            bidderinfos_without_w = bidderInfos.copy(); del bidderinfos_without_w[w]
            revenue_without_w = self._calculate_step(slots, bidderInfos, bidderinfo_winner, bidderinfos_without_w)
            prices_vcg[w] = bidderinfo_winner.budget - max(0,(revenue_raw-revenue_without_w))
        revenue_vcg = sum(prices_vcg.itervalues())
        logging.info('vcg:\trevenue %d\tprices: %s' % (revenue_vcg,sorted(prices_vcg.iteritems())))
        return (revenue_vcg,prices_vcg)
    
    def _calculate_step(self, slots, bidderInfos, bidderinfo_winner, bidderinfos_without_w):
        prob_vcg, _prob_vars = self.gwd.generate(slots,bidderinfos_without_w)
        prob_vcg.name = 'vcg_%d' % bidderinfo_winner.id
        logging.info('vcg:\tcalculating - without winner %s' % (bidderinfo_winner.id,))
        solver_status = prob_vcg.solve(self.gwd.solver)
        assert solver_status == pu.LpStatusOptimal
        
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
        epsilon = max(1,sum(prices_vcg)-1)
        # ebpo objective function
        prob_ebpo += sum(pi.itervalues()) + epsilon*m
        # ebpo constraint: π_j - m <= π_j^vcg 
        for w in winners:
            prob_ebpo += pi[w]-m <= prices_vcg[w]
        ebpo_solver = pu.GUROBI(msg=False, mip=False)
        
        # initialize revenue_sep
        revenue_last_sep = 0
        # abort after 1000 iterations (this should never happen)
        for cnt in xrange(1000):
            # make a copy of prob_gwd, since we are using it as basis
            prob_sep = prob_gwd.deepcopy()
            prob_sep.name = 'sep'
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
            # assert solver_status == pu.LpStatusOptimal
            # save the value: z( π^t )
            revenue_sep = pu.value(prob_sep.objective)
            
            # check for a blocking coalition. if no coalition exists, break 
            blocking_coalition_exists = revenue_sep > sum(prices_t.itervalues())
            if not blocking_coalition_exists: break
            
            # if the z(π^t) did not change between iterations, we are in a local optimum we cannot escape
            if revenue_sep == revenue_last_sep: break
            else: revenue_last_sep = revenue_sep
            
            # extend and solve the ebpo problem:
            blocking_coalition = frozenset(int(b.name[2:]) for b in prob_sep.variables() if b.varValue==1 and b.name[:1]=='y')
            logging.info('sep:\tvalue: %d, blocking: %s' % (revenue_sep, sorted(blocking_coalition),))
            
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
            
            # update the π_t list. π_t has to be equal or increase for each t
            prices_t = dict(
#                (int(b.name[3:]), max(b.varValue,prices_t[int(b.name[3:])])) 
                (int(b.name[3:]), b.varValue) 
                for b in prob_ebpo.variables() 
                if b.name[:2]=='pi'
            )
            logging.info('ebpo:\trevenue: %d' % (sum(prices_t.itervalues()),))
        else:
            raise Exception('too many iterations in core calculation')
        # there is no blocking coalition -> the current iteration of the ebpo contains core prices
        revenue_core = sum(prices_t.itervalues())
        logging.info('core:\trevenue %d\tprices: %s' % (revenue_core,prices_t.items()))
        return (revenue_core, prices_t)

class TvAuctionProcessor(object):
    def __init__(self):
        self.gwdClass = Gwd
        self.vcgClass = Vcg
        self.coreClass = CorePricing
    def isOptimal(self,solver_status):
        return solver_status == pu.LpStatusOptimal
    
    def solve(self, slots, bidderInfos, timeLimit=15, **kw):
    
        # vars needed for the heuristic
        bidders_satisfied = []
        
        # generate the gwd    
        gwd = self.gwdClass()
        prob_gwd, _prob_vars = gwd.generate(slots, bidderInfos)
        
        # solve the gwd with a time limit to determine whether a heuristic has to be used
        gwd.solver.timeLimit = timeLimit
        gwd.solver.epgap = 0.02
        
        _solver_status, (revenue_raw, prices_raw, winners) = gwd.solve(prob_gwd, bidderInfos)

        # solve vcg
        vcg = self.vcgClass(gwd)
        _revenue_vcg, prices_vcg = vcg.calculate(slots, bidderInfos, revenue_raw, winners, bidders_satisfied)
        
        core = self.coreClass(gwd)
        _revenue_core, prices_core = core.solve(prob_gwd, bidderInfos, winners, prices_raw, prices_vcg)
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
    parser.add_option('-l','--log', dest='loglevel', help='the log level', default='WARN')
    parser.add_option('-s','--scenario', dest='scenario', help='the scenario')
    options = parser.parse_args()[0]
    numeric_level = getattr(logging, options.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % options.loglevel)
    if not options.scenario:
        raise ValueError('Scenario needed')
    
    logging.basicConfig(level=numeric_level)
    
    slots, bidderInfos = json.loads(options.scenario)
    res = TvAuctionProcessor.solve(slots, bidderInfos)
    
    print json.dumps(res,indent=2)    
