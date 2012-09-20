# simluation 


import random
from collections import defaultdict
import numpy as np
from scipy import stats
from pprint import pprint as pp

FIXED = 0
LINEAR = 1

CONSTANT = 1
NORMAL = 2
UNIFORM = 3

FIXED = 'fixed'
LINEAR = 'linear'

CONSTANT = 'constant'
NORMAL = 'normal'
UNIFORM = 'uniform'

def minmax(data,minval,maxval):
    return max(min(data,maxval),minval)
 
class SimulationComponent(object):
    def __init__(self, slot_qty, bidder_qty):
        self.slot_qty = slot_qty
        self.bidder_qty = bidder_qty
    def getData(self, stype):
        raise NotImplementedError('implement this')
    
    def _findNearest(self,ar,val):
        idx = (np.abs(ar-val)).argmin()
        return ar[idx]
    
    def _getDataContinuos(self,stype,val_max,qty):
        half = val_max/2
        quarter = val_max/4
        if stype==CONSTANT:
            return [half]*qty
        elif stype==UNIFORM:
            return [random.randint(0,val_max) for _i in xrange(qty)]
        elif stype==NORMAL:
            return [minmax(int(random.gauss(half,quarter)), 0, val_max) for _i in xrange(qty)]
        

class SlotDuration(SimulationComponent):
    def __init__(self, slot_duration_max, *a, **kw):
        super(SlotDuration, self).__init__(*a, **kw)
        self.slot_duration_max = slot_duration_max
    def getData(self, stype):
        return self._getDataContinuos(stype, self.slot_duration_max, self.slot_qty)
    
class SlotPricePerSecond(SimulationComponent):
    def __init__(self, slot_price_steps, *a, **kw):
        super(SlotPricePerSecond, self).__init__(*a, **kw)
        self.slot_price_steps = np.array(slot_price_steps)
    def getData(self, stype):
        if stype==CONSTANT:
            return [self.slot_price_steps[0]]*self.slot_qty
        elif stype==UNIFORM:
            return [random.choice(self.slot_price_steps) for _i in xrange(self.slot_qty)]
        elif stype==NORMAL:
            return [
                self._findNearest(self.slot_price_steps, stats.scoreatpercentile(
                    self.slot_price_steps,
                    minmax(int(random.gauss(50,25)),0,100)
                ))
                for _i in xrange(self.slot_qty)
            ]
class SlotPriceToPrio(SimulationComponent):
    @staticmethod
    def _prioFixed(slot_price):
        return 100
    @staticmethod
    def _prioLinear(slot_price):
        return slot_price
    
    def __init__(self, slot_prices, *a, **kw):
        super(SlotPriceToPrio, self).__init__(*a, **kw)
        self.slot_prices = slot_prices
        self.prio_functions = {
            FIXED: SlotPriceToPrio._prioFixed,
            LINEAR: SlotPriceToPrio._prioLinear
        }
        
    def getData(self, stype, prio_type):
        prio_fn = self.prio_functions[prio_type]
        res = {}
        for slot_type, slot_type_prices in self.slot_prices.iteritems():
            slot_prios = [prio_fn(slot_price) for slot_price in slot_type_prices]
            if stype==CONSTANT:
                res[slot_type] = slot_prios
            elif stype==UNIFORM:
                res[slot_type] = [random.randint(0,slot_prio*2) for slot_prio in slot_prios]
            elif stype==NORMAL:
                res[slot_type] = [int(random.gauss(slot_prio,slot_prio/2)) for slot_prio in slot_prios]
        return res
    
class AdvertBidder(SimulationComponent):
    def __init__(self, advert_duration_max, *a, **kw):
        super(AdvertBidder, self).__init__(*a, **kw)
        self.advert_duration_max = advert_duration_max
    def getData(self, stype):
        return self._getDataContinuos(stype, self.advert_duration_max, self.slot_qty)
    
   
def main(slot_qty,bidder_qty,slot_duration_max,advert_duration_max,slot_price_steps,advert_price_max):
    '''starts the main simulation
    for every possible vector, all combinations are tested.'''
    
    # set the seed to random in order to get consistent results
    random.seed(1)
    
    # target vectors:
    
    # slot duration (constant, normal, uniform)
    slotDuration = SlotDuration(slot_duration_max, slot_qty, bidder_qty)
    slot_durations = {}
    for stype in (CONSTANT,NORMAL,UNIFORM):
        slot_durations[stype] = slotDuration.getData(stype)
    
    # slot price per second (constant, normal, uniform)
    slotPricePerSecond = SlotPricePerSecond(slot_price_steps, slot_qty, bidder_qty)
    slot_prices = {}
    for stype in (CONSTANT,NORMAL,UNIFORM):
        slot_prices[stype] = slotPricePerSecond.getData(stype)
    
    # priority vector depending on slot price (constant, linear, uniform)
    # fixed_constant: all priorities are the same, independent of the price set
    # fixed_uniform: all priorities are uniformely distributed, independent of the price set
    # fixed_normal: all priorities are normally distributed, independent of the price set
    # linear_constant: all priorities are linearly correlated to the price set
    # linear_normal: all priorities are linearly correlated to the price set and normally varied between bidders
    # linear_uniform: all priorities are linearly correlated to the price set and uniformly varied between bidders (+- 100%)
    
    priority_bidders = {}
    slotPriceToPrio = SlotPriceToPrio(slot_prices, slot_qty, bidder_qty)
    
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in priority_bidders: priority_bidders[bidder_id] = {}
        for stype in (CONSTANT,NORMAL,UNIFORM):
            if stype not in priority_bidders[bidder_id]: priority_bidders[bidder_id][stype] = {}
            for prio_type in slotPriceToPrio.prio_functions:
                priority_bidders[bidder_id][stype][prio_type] = slotPriceToPrio.getData(stype, prio_type)
    # this either
    # priority vector dependent to slot price (correlated, semicorrelated, uncorrelated)
    # priority vectors between bidders (constant, normal, uniform)
    pp(priority_bidders)
    
    # advert duration (constant, normal, uniform)
    advert_bidders = {}
    advertBidder = AdvertBidder(advert_duration_max, slot_qty, bidder_qty)
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in advert_bidders: advert_bidders[bidder_id] = {}
        for stype in (CONSTANT,NORMAL,UNIFORM):
            advert_bidders[bidder_id][stype] = advertBidder.getData(stype)
            
if __name__=='__main__':
    slot_qty = 4
    bidder_qty = 2
    slot_duration_max = 120 
    advert_duration_max = 120
    slot_price_steps = [1.0,2.0,5.0,10.0,20.0,50.0,75.0]
    advert_price_max = 100.0
    
    main(slot_qty,bidder_qty,slot_duration_max,advert_duration_max,slot_price_steps,advert_price_max)
    
    
    