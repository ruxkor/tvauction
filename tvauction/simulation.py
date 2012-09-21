# -*- coding: utf-8; -*-
# simulation 


import random
import numpy as np
from scipy import stats
from pprint import pprint as pp

import processor_pulp
import logging

import matplotlib.pyplot as plt

FIXED = 0
LINEAR = 1

CONSTANT = 1
NORMAL = 2
UNIFORM = 3

#FIXED = 'fixed'
#LINEAR = 'linear'
#
#CONSTANT = 'constant'
#NORMAL = 'normal'
#UNIFORM = 'uniform'

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
                    minmax(random.gauss(50,25),0,100)
                ))
                for _i in xrange(self.slot_qty)
            ]
class SlotPriceToPrio(SimulationComponent):
    @staticmethod
    def _prioFixed(slot_price):
        return 10.0
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
                res[slot_type] = [round(max(0,random.gauss(slot_prio,slot_prio/2)),2) for slot_prio in slot_prios]
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
        return cmin+self._getDataContinuos(stype, cmax-cmin, 1)[0]
    
def generate(slot_qty,bidder_qty,slot_duration_max,advert_duration_max,slot_price_steps,advert_price_max,campaign_min_prio_range):
    '''starts the main simulation generation for every possible vector.'''
    
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
    
    # advert duration (constant, normal, uniform)
    advert_durations = {}
    advertDuration = AdvertDuration(advert_duration_max, slot_qty, bidder_qty)
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in advert_durations: advert_durations[bidder_id] = {}
        for stype in (CONSTANT,NORMAL,UNIFORM):
            advert_durations[bidder_id][stype] = advertDuration.getData(stype)
    
    # advert price (constant, normal, uniform)
    advert_prices = {}
    advertPrice = AdvertPrice(advert_price_max, slot_qty, bidder_qty)
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in advert_prices: advert_prices[bidder_id] = {}
        for stype in (CONSTANT,NORMAL,UNIFORM):
            advert_prices[bidder_id][stype] = advertPrice.getData(stype)
    
    campaign_min_prio_sum = {}
    campaignMinPrioSum = CampaignMinPrioSum(campaign_min_prio_range, slot_qty, bidder_qty)
    for bidder_id in xrange(bidder_qty):
        if bidder_id not in campaign_min_prio_sum: campaign_min_prio_sum[bidder_id] = {}
        for stype in (CONSTANT,NORMAL,UNIFORM):
            campaign_min_prio_sum[bidder_id][stype] = campaignMinPrioSum.getData(stype)
#    pp(campaign_min_prio_sum)
    
    return (slot_durations, slot_prices, priority_bidders, advert_durations, advert_prices, campaign_min_prio_sum)


def drawResult(file_prefix, res):

    fig = plt.figure(None,figsize=(20,6))
    
    # the first graph shows all winners and how much they paid
    ind = np.array(sorted(res['winners']))
    width = 0.8
    
    ax1 = fig.add_subplot(111)
    ax1.grid(True,axis='y')
    
    bars = []
    bar_labels = []
    
    for (nr,(ptype,pcolor)) in enumerate(zip(['raw','vcg','core','final'],[(0,1,1),(0,0,0),(1,0,1),(1,1,0)])):
        bar_width = width - nr*width*0.2
        vals = [v for (k,v) in sorted(res['prices_%s' % ptype].iteritems()) if k in ind]
        bar = ax1.bar(ind-bar_width*0.5, vals, bar_width, color=pcolor,linewidth=0.5)
        bars.append(bar)
        bar_labels.append(ptype)

    ax1.set_xticks(ind)
    ax1.set_xticklabels(ind)
    ax1.legend(bars,bar_labels)
    
    fig.savefig(file_prefix+'_1.pdf')

    # the second graph is the step info graph    
    steps_info = res['step_info']
    fig = plt.figure(None,figsize=(16,9))
    ax2 = fig.add_subplot(111)
    ax2.grid(True,axis='y')
    
    step_max = 0
    tuples_all = {}
    for what in ('raw','vcg','sep','blocking_coalition','ebpo'):
        tuples_all[what] = tuples_what = [(nr, step_info[what]) for (nr, step_info) in enumerate(steps_info) if what in step_info]
        step_max = max(step_max, *(s for (s,v) in tuples_what))
        
    for what in ('raw','vcg','sep','blocking_coalition','ebpo'):
        tuples_what = tuples_all[what]
        if tuples_what[-1][0] != step_max:
            tuples_what.append( (step_max,None) )
        
        steps_what = []
        values_what = []
        for nr, (step_what, value_what) in enumerate(tuples_what):
            if steps_what:
                step_prev, value_prev = tuples_what[nr-1]
                for i in range(step_prev+1,step_what):
                    steps_what.append(i)
                    values_what.append(value_prev)
            steps_what.append(step_what)
            values_what.append(value_what if value_what is not None else value_prev)
        
        ax2.plot(steps_what, values_what,drawstyle='steps-post',label=what,linestyle='-', linewidth=2.0, alpha=0.8)
    
    
    ax2.set_xlim(-0.1,step_max+0.1)
    ax2.legend(loc='upper left')
    fig.savefig(file_prefix+'_2.pdf')
    
if __name__=='__main__':
    from optparse import OptionParser
    import sys
    import json
    def convertToJson(opt, opt_str, value, parser):
        setattr(parser.values,opt.dest,json.loads(value))
    parser = OptionParser()
    parser.add_option('--random-seed',dest='random_seed',type='int',default=1,help='random seed')
    parser.add_option('--zero-price-vector',dest='zero_price',action='store_true',default=False,help='don\'t calculate the vcg prices. use a 0 price vector instead')
    parser.add_option('--draw',dest='draw_results',action='store_true',default=False,help='draw graph depicting the allocation and the process')
    parser.add_option('--slot-qty',dest='slot_qty',type='int',default=50,help='slot quantity')
    parser.add_option('--bidder-qty',dest='bidder_qty',type='int',default=50,help='bidder quantity')
    parser.add_option('--slot-duration-max',dest='slot_duration_max',type='int',default=120,help='slot maximum duration')
    parser.add_option('--advert-duration-max',dest='advert_duration_max',type='int',default=100,help='advert maximum duration')
    parser.add_option('--advert-price-max',dest='advert_price_max',type='float',default=100.0,help='advert maximum price (per second)')
    parser.add_option(
            '--slot-price-steps',dest='slot_price_steps',type='str',action='callback',default=[1.0,2.0,5.0,10.0,20.0,50.0,75.0],
            help='slot price (per second), in steps [json]',callback=convertToJson
    )
    parser.add_option(
            '--campaign-min-prio-range',dest='campaign_min_prio_range',type='str',action='callback',default=[1,50],
            help='campaign minimum priority vector sum ranges (min/max, as percent) [json]', callback=convertToJson
    )
    parser.add_option(
            '--distributions',dest='distributions',type='str',action='callback',callback=convertToJson,
            default=[
#                [CONSTANT,CONSTANT,CONSTANT,CONSTANT,CONSTANT,CONSTANT,FIXED],
                [NORMAL,NORMAL,NORMAL,NORMAL,NORMAL,NORMAL,LINEAR],
            ],
            help='distributions for the following values:'
                    'slot duration (cnu), advert duration (cnu), '
                    'slot reserve price (cnu), bidder\'s ad price (cnu), ' 
                    'minimum prio vector (cnu), inter-bidding priority vectors (cnu),'
                    'relation priority vector to price (fl).' 
                    'values as a list of lists [json]'
    )
    options, args = parser.parse_args(sys.argv)

    slot_qty = options.slot_qty
    bidder_qty = options.bidder_qty
    slot_duration_max = options.slot_duration_max 
    advert_duration_max = options.advert_duration_max
    advert_price_max = options.advert_price_max
    campaign_min_prio_range = options.campaign_min_prio_range
    slot_price_steps = options.slot_price_steps
    distributions = options.distributions
    
    random_seed = options.random_seed
    draw_results = options.draw_results or True
    use_zero_price = options.zero_price
    
    # set the seed to random in order to get consistent results
    if random_seed: random.seed(random_seed)
    
    # 1.    define quantities
    #        - slot quantity (int)
    #        - bidder quantity (int)
    #        - slot maximum duration (int)
    #        - slot price (per second), in steps (list[float])
    #        - advert maximum duration (int)
    #        - advert maximum price (per second), float
    #        - campaign minimum priority vector sum ranges (min/max, as percent), (int,int)
    # 2.    define reserve price distribution
    #        - constant: all slots have the same price per second
    #        - normal: the prices are normally distributed
    #        - uniform: the prices are uniformly distributed
    # 3.    define bidding behavior
    # 3.1.    - define distribution between priorities
    #            - constant: all bidders will have the same priority vector
    #            - normal: the differences in the priority vector will be normally distributed
    #            - uniform: the differences in the priority vector will be uniformly distributed
    # 3.2    - define relation between reserve price and priority vector
    #            - fixed: there is no correlation between the reserve price and the priority vector
    #            - linear: there is a linear correlation between the reserve price and the priority vector
    # 4.    define slot duration distribution
    #        - constant
    #        - normal
    #        - uniform
    # 5.    define advert duration distribution
    #        - constant
    #        - normal
    #        - uniform
    # 6.    define advert price (per second) distribution
    #        - constant
    #        - normal
    #        - uniform
    # 7.    define the minimum priority vector sum (as a fraction) distribution
    #        - constant
    #        - normal
    #        - uniform
    #
    # note:
    #    the final campaign price will be calculated using the following formula for each bidder:
    
    
    slot_durations, slot_prices, priority_bidders, advert_durations, advert_prices, campaign_min_prio_sum = \
        generate(
             slot_qty,
             bidder_qty,
             slot_duration_max,
             advert_duration_max,
             slot_price_steps,
             advert_price_max,
             campaign_min_prio_range
        )


    
    logging.basicConfig(level=logging.INFO)
    auction_processor = processor_pulp.TvAuctionProcessor()
    if use_zero_price: auction_processor.vcgClass = processor_pulp.VcgFake
    
    results = []
    for distribution in distributions:
        d_slot_duration, d_ad_duration, d_slot_price, d_bid_price, d_min_prio, d_inter_prio, d_prio_to_price = distribution
        # generate slot objects
        # generate campaign prices
        # generate bidderInfo object
        # solve!
        slots = dict(
            (slot_id,processor_pulp.Slot(
                id=slot_id,
                price=slot_prices[d_slot_price][slot_id],
                length=slot_durations[d_slot_duration][slot_id]
            ))
            for slot_id in xrange(slot_qty)
        )
        
        # campaign prices are defined as follows:
        # bidder_attrib_min = Σ(bidder_prio_values) * campaign_prio_sum / 100
        # campaign_budget = advert_price * ad_duration * bidder_attrib_min
        bidderInfos = {}
        for bidder_id in xrange(bidder_qty):
            bidder_prio_values = dict(enumerate(priority_bidders[bidder_id][d_inter_prio][d_prio_to_price][d_slot_price]))
            bidder_ad_duration = advert_durations[bidder_id][d_ad_duration]
            bidder_attrib_min = int(sum(bidder_prio_values.itervalues()) * campaign_min_prio_sum[bidder_id][d_min_prio] / 100)
            bidderInfos[bidder_id] = processor_pulp.BidderInfo(
                id=bidder_id,
                budget=advert_prices[bidder_id][d_bid_price] * bidder_ad_duration * bidder_attrib_min,
                length=bidder_ad_duration,
                attrib_min=bidder_attrib_min,
                attrib_values=bidder_prio_values
            )
            
        res = auction_processor.solve(slots, bidderInfos,5,20)
        if draw_results:
            now = '' # datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            drawResult('/tmp/tvauction_simulation_%s_%s' % (now,''.join(map(str,distribution))), res)
        results.append(res)
