#!/usr/bin/env python
# -*- coding: utf-8; -*-
# generates bidder_info and slot objects

import os
import sys
import random
import datetime
import time
import logging

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from tvauction.common import Slot, BidderInfo

CONSTANT = 0
UNIFORM = 1
NORMAL = 2
NORMAL_NARROW = 3

FIXED = 0
LINEAR = 1
QUADRATIC = 2
SQUARE_ROOT = 3

def minmax(data,minval,maxval):
    return max(min(data,maxval),minval)
 
class SimulationComponent(object):
    def __init__(self, slot_qty, bidder_qty, bid_qty):
        self.slot_qty = slot_qty
        self.bidder_qty = bidder_qty
        self.bid_qty = bid_qty
        
    def getData(self, stype):
        raise NotImplementedError('implement this')
    
    def _findNearest(self, ar, val):
        idx = (np.abs(ar-val)).argmin()
        return ar[idx]
    
    def _getDataContinuos(self, stype, val_max, qty):
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
                    minmax(random.gauss(50,25),0,100)
                ))
                for _i in xrange(self.slot_qty)
            ]
class SlotPriceToPrio(SimulationComponent):
    @staticmethod
    def _prioFixed(slot_price):
        return 10
    @staticmethod
    def _prioLinear(slot_price):
        return int(2*slot_price)
    @staticmethod
    def _prioQuadratic(slot_price):
        return int((2*slot_price)**2)
    @staticmethod
    def _prioSqrt(slot_price):
        return int((2*slot_price)**0.5)
    
    def __init__(self, slot_prices, *a, **kw):
        super(SlotPriceToPrio, self).__init__(*a, **kw)
        self.slot_prices = slot_prices
        self.prio_functions = {
            FIXED: SlotPriceToPrio._prioFixed,
            LINEAR: SlotPriceToPrio._prioLinear,
            QUADRATIC: SlotPriceToPrio._prioQuadratic,
            SQUARE_ROOT: SlotPriceToPrio._prioSqrt
        }
        
    def getData(self, stype, prio_type):
        prio_fn = self.prio_functions[prio_type]
        res = {}
        for slot_type, slot_type_prices in self.slot_prices.iteritems():
            slot_prios = (prio_fn(slot_price) for slot_price in slot_type_prices)
            if stype==CONSTANT:
                res[slot_type] = slot_prios
            elif stype==UNIFORM:
                res[slot_type] = [random.randint(0,slot_prio) for slot_prio in slot_prios]
            elif stype==NORMAL:
                res[slot_type] = [minmax(int(random.gauss(slot_prio*0.5,slot_prio*0.25)), 0, slot_prio) for slot_prio in slot_prios]
            elif stype==NORMAL_NARROW:
                res[slot_type] = [minmax(int(random.gauss(slot_prio*0.75,slot_prio*0.125)), 0, slot_prio) for slot_prio in slot_prios]
        return res
    
class AdvertDuration(SimulationComponent):
    def __init__(self, advert_duration_max, *a, **kw):
        super(AdvertDuration, self).__init__(*a, **kw)
        self.advert_duration_max = advert_duration_max
    def getData(self, stype):
        return self._getDataContinuos(stype, self.advert_duration_max, 1)[0]
    
class AdvertPrice(SimulationComponent):
    def __init__(self, advert_price_max, *a, **kw):
        super(AdvertPrice, self).__init__(*a, **kw)
        self.advert_price_max = advert_price_max
    def getData(self, stype):
        return self._getDataContinuos(stype, self.advert_price_max, 1)[0]

class CampaignMinPrioSum(SimulationComponent):
    def __init__(self, campaign_min_prio_range, *a, **kw):
        super(CampaignMinPrioSum, self).__init__(*a, **kw)
        self.campaign_min_prio_range = campaign_min_prio_range
    def getData(self, stype):
        cmin, cmax = campaign_min_prio_range
        # data gets already trimmed to boundaries
        return cmin+self._getDataContinuos(stype, cmax-cmin, 1)[0]
    
def pregenerate(slot_qty,bidder_qty,bid_qty,slot_duration_max,advert_duration_max,slot_price_steps,advert_price_max,campaign_min_prio_range):
    '''starts the main simulation generation for every possible vector.'''
    
    # target vectors:
    
    # slot duration (constant, normal, uniform)
    slotDuration = SlotDuration(slot_duration_max, slot_qty, bidder_qty, bid_qty)
    slot_durations = {}
    for stype in (CONSTANT,NORMAL,UNIFORM):
        slot_durations[stype] = slotDuration.getData(stype)
    
    # slot price per second (constant, normal, uniform)
    slotPricePerSecond = SlotPricePerSecond(slot_price_steps, slot_qty, bidder_qty, bid_qty)
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
    slotPriceToPrio = SlotPriceToPrio(slot_prices, slot_qty, bidder_qty, bid_qty)
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in priority_bidders: priority_bidders[bidder_id] = {}
        for stype in (CONSTANT,UNIFORM,NORMAL,NORMAL_NARROW):
            if stype not in priority_bidders[bidder_id]: priority_bidders[bidder_id][stype] = {}
            for prio_type in slotPriceToPrio.prio_functions:
                priority_bidders[bidder_id][stype][prio_type] = slotPriceToPrio.getData(stype, prio_type)
    
    # advert duration (constant, normal, uniform)
    advert_durations = {}
    advertDuration = AdvertDuration(advert_duration_max, slot_qty, bidder_qty, bid_qty)
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in advert_durations: advert_durations[bidder_id] = {}
        for stype in (CONSTANT,NORMAL,UNIFORM):
            advert_durations[bidder_id][stype] = advertDuration.getData(stype)
    
    # advert price (constant, normal, uniform)
    advert_prices = {}
    advertPrice = AdvertPrice(advert_price_max, slot_qty, bidder_qty, bid_qty)
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in advert_prices: advert_prices[bidder_id] = {}
        for stype in (CONSTANT,NORMAL,UNIFORM):
            advert_prices[bidder_id][stype] = tuple(advertPrice.getData(stype) for i in range(bid_qty))
    
    campaign_min_prio_sum = {}
    campaignMinPrioSum = CampaignMinPrioSum(campaign_min_prio_range, slot_qty, bidder_qty, bid_qty)
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in campaign_min_prio_sum: campaign_min_prio_sum[bidder_id] = {}
        for stype in (CONSTANT,NORMAL,UNIFORM):
            campaign_min_prio_sum[bidder_id][stype] = tuple(campaignMinPrioSum.getData(stype) for i in range(bid_qty))
            
    return (slot_durations, slot_prices, priority_bidders, advert_durations, advert_prices, campaign_min_prio_sum)

def generateScenario(pregen_data):
    slot_durations, slot_prices, priority_bidders, advert_durations, advert_prices, campaign_min_prio_sum = pregen_data
    
    d_slot_duration, d_ad_duration, d_slot_price, d_bid_price, d_min_prio, d_inter_prio, d_prio_to_price = distribution
    # generate slot objects
    slots = dict(
        (slot_id, Slot(
            id=slot_id,
            price=slot_prices[d_slot_price][slot_id],
            length=slot_durations[d_slot_duration][slot_id]
        ))
        for slot_id in xrange(slot_qty)
    )
    
    # generate bidder_info objects
    # 
    # campaign prices are defined as follows:
    # bidder_attrib_min = Σ(bidder_prio_values) * campaign_prio_sum / 100
    # campaign_budget = advert_price * ad_duration * bidder_attrib_min
    bidder_infos = {}
    for bidder_id in xrange(bidder_qty):
        bidder_prio_values = dict(enumerate(priority_bidders[bidder_id][d_inter_prio][d_prio_to_price][d_slot_price]))
        bidder_ad_duration = advert_durations[bidder_id][d_ad_duration]
        
        bidder_bids = []
        for i in range(bid_qty):
            bidder_attrib_min = int(sum(bidder_prio_values.itervalues()) * campaign_min_prio_sum[bidder_id][d_min_prio][i] / 100)
            bidder_bid_price = advert_prices[bidder_id][d_bid_price][i] * bidder_ad_duration * bidder_attrib_min
            bidder_bids.append((bidder_bid_price,bidder_attrib_min))
        bidder_infos[bidder_id] = BidderInfo(
            id=bidder_id,
            length=bidder_ad_duration,
            bids=tuple(bidder_bids),
            attrib_values=bidder_prio_values
        )
    return (slots, bidder_infos) 
    
if __name__=='__main__':
    import optparse
    
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    from tvauction.common import json

    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    logging.basicConfig(level=log_level)
    
    def convertJson(opt, opt_str, value, parser): setattr(parser.values, opt.dest, json.decode(value))
    parser = optparse.OptionParser()
    parser.add_option('--slot-qty', dest='slot_qty', type='int', default=336, help='slot quantity')
    parser.add_option('--bidder-qty', dest='bidder_qty', type='int', default=50, help='bidder quantity')
    parser.add_option('--bid-qty', dest='bid_qty', type='int', default=2, help='bid quantity')
    parser.add_option('--slot-duration-max', dest='slot_duration_max', type='int', default=120, help='slot maximum duration')
    parser.add_option('--advert-duration-max', dest='advert_duration_max', type='int', default=40, help='advert maximum duration')
    parser.add_option('--advert-price-max', dest='advert_price_max', type='float', default=120.0, help='advert maximum price (per second)')
    parser.add_option(
        '--random-seeds', dest='random_seeds', type='str', action='callback', callback=convertJson,
        help='random seeds. if a falsy value is passed, a random seed will be used. [json]', default=[1,2,3]
    )
    parser.add_option(
        '--slot-price-steps', dest='slot_price_steps', type='str', action='callback', 
        default=[1.0,2.0,5.0,10.0,20.0,50.0,75.0],
        help='slot price (per second), in steps [json]', callback=convertJson
    )
    parser.add_option(
        '--campaign-min-prio-range', dest='campaign_min_prio_range', type='str', action='callback', default=[10,50],
        help='campaign minimum priority vector sum ranges (min/max, as percent) [json]', callback=convertJson
    )
    parser.add_option(
        '--distributions', dest='distributions', type='str', action='callback', callback=convertJson,
        default=[
#            [CONSTANT,CONSTANT,CONSTANT,CONSTANT,CONSTANT,CONSTANT,FIXED],
            [NORMAL,NORMAL,NORMAL,NORMAL,NORMAL,NORMAL,LINEAR],
#            [CONSTANT,NORMAL,NORMAL,NORMAL,NORMAL,NORMAL_NARROW,LINEAR],
        ],
        help=   'distributions for the following values:'
                'slot duration (cnu), advert duration (cnu), '
                'slot reserve price (cnu), bidder\'s ad price (cnu), ' 
                'minimum prio vector (cnu), inter-bidding priority vectors (cnun),'
                'relation priority vector to price (flqs).' 
                'values as a list of lists [json]'
    )
    for option in parser.option_list: 
        if option.default != ("NO", "DEFAULT"): option.help += (" " if option.help else "") + "[default: %default]"
    
    options, args = parser.parse_args(sys.argv)

    slot_qty = options.slot_qty
    bidder_qty = options.bidder_qty
    bid_qty = options.bid_qty
    slot_duration_max = options.slot_duration_max 
    advert_duration_max = options.advert_duration_max
    advert_price_max = options.advert_price_max
    campaign_min_prio_range = options.campaign_min_prio_range
    slot_price_steps = options.slot_price_steps
    distributions = options.distributions
    
    random_seeds = options.random_seeds or [None]
    
    # start generating result
    res = {
        'options': options.__dict__,
        'scenarios': []
    }
    
    # set the seed to random in order to get consistent results
    for random_seed in random_seeds:
        random.seed(random_seed)
    
        pregen_data = pregenerate(
             slot_qty,
             bidder_qty,
             bid_qty,
             slot_duration_max,
             advert_duration_max,
             slot_price_steps,
             advert_price_max,
             campaign_min_prio_range
        )
    
        for distribution in distributions:
            res_dist = generateScenario(pregen_data)
            res['scenarios'].append(res_dist)
    
    # print scenarios to stdout
    print json.encode(res['scenarios'])
    # and options for reference to stderr
    print >> sys.stderr, json.encode(res['options'])    
