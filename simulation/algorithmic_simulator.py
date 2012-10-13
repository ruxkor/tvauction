#!/usr/bin/env python
# -*- coding: utf-8; -*-
# simulation 

import os
import sys
import random
import datetime
import time
import logging

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import tvauction.processor

from pprint import pprint as pp

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
 
if __name__=='__main__':
    import optparse
    import json
    
    def convertToJson(opt, opt_str, value, parser):
        setattr(parser.values,opt.dest,json.loads(value))
    parser = optparse.OptionParser()
    parser.add_option('--random-seed',dest='random_seed',type='int',default=1,help='random seed')
    parser.add_option('--initial-vector',dest='price_vector',choices=('vcg','zero'),default='vcg',help='the type of price vector used as a starting point for core price generation (vcg,zero)')
    parser.add_option('--core-algorithm',dest='core_algorithm',choices=('trim','switch','reuse'),default='reuse',help='which algorithm should be used in case a suboptimal winner determination is discovered during core pricing (trim: trim the values to be within a feasible region, switch: recreate the ebpo,reuse: recreate the ebpo and try to re-use already existing constraints)')
    parser.add_option('--no-draw',dest='draw_results',action='store_false',default=True,help='draw graphs illustrating the allocation and the process')
    parser.add_option('--no-solve',dest='solve_problems',action='store_false',default=True,help='draw graphs illustrating the allocation and the process')
    parser.add_option('--slot-qty',dest='slot_qty',type='int',default=20,help='slot quantity')
    parser.add_option('--bidder-qty',dest='bidder_qty',type='int',default=40,help='bidder quantity')
    parser.add_option('--slot-duration-max',dest='slot_duration_max',type='int',default=120,help='slot maximum duration')
    parser.add_option('--advert-duration-max',dest='advert_duration_max',type='int',default=40,help='advert maximum duration')
    parser.add_option('--advert-price-max',dest='advert_price_max',type='float',default=120.0,help='advert maximum price (per second)')
    parser.add_option(
            '--slot-price-steps',dest='slot_price_steps',type='str',action='callback',default=[1.0,2.0,5.0,10.0,20.0,50.0,75.0],
            help='slot price (per second), in steps [json]',callback=convertToJson
    )
    parser.add_option(
            '--campaign-min-prio-range',dest='campaign_min_prio_range',type='str',action='callback',default=[15,50],
            help='campaign minimum priority vector sum ranges (min/max, as percent) [json]', callback=convertToJson
    )
    parser.add_option(
            '--distributions',dest='distributions',type='str',action='callback',callback=convertToJson,
            default=[
#                [CONSTANT,CONSTANT,CONSTANT,CONSTANT,CONSTANT,CONSTANT,FIXED],
                [NORMAL,NORMAL,NORMAL,NORMAL,NORMAL,NORMAL,LINEAR],
                [CONSTANT,NORMAL,NORMAL,NORMAL,NORMAL,NORMAL_NARROW,LINEAR],
            ],
            help=   'distributions for the following values:'
                    'slot duration (cnu), advert duration (cnu), '
                    'slot reserve price (cnu), bidder\'s ad price (cnu), ' 
                    'minimum prio vector (cnu), inter-bidding priority vectors (cnun),'
                    'relation priority vector to price (flqs).' 
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
    draw_results = options.draw_results
    solve_problems = options.solve_problems
    price_vector = options.price_vector
    core_algorithm = options.core_algorithm
    
    # set the seed to random in order to get consistent results
    if random_seed: random.seed(random_seed)
    
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

    
    log_level = int(os.environ['LOG_LEVEL']) \
        if 'LOG_LEVEL' in os.environ \
        else logging.WARN
    if log_level < logging.INFO:
        tvauction.processor.SOLVER_MSG = True
    
    logging.basicConfig(level=log_level)
    
    auction_processor = tvauction.processor.TvAuctionProcessor()
    
    if price_vector=='vcg': auction_processor.vcgClass = tvauction.processor.Vcg
    elif price_vector=='zero': auction_processor.vcgClass = tvauction.processor.Zero
    
    if core_algorithm=='trim': auction_processor.core_algorithm = tvauction.processor.CorePricing.TRIM_VALUES
    elif core_algorithm=='switch': auction_processor.core_algorithm = tvauction.processor.CorePricing.SWITCH_COALITIONS
    elif core_algorithm=='reuse': auction_processor.core_algorithm = tvauction.processor.CorePricing.REUSE_COALITIONS
    
    for distribution in distributions:
        print ''
        print '-' * 40
        print 'distribution: ', distribution
        d_slot_duration, d_ad_duration, d_slot_price, d_bid_price, d_min_prio, d_inter_prio, d_prio_to_price = distribution
        # generate slot objects
        slots = dict(
            (slot_id,tvauction.processor.Slot(
                id=slot_id,
                price=slot_prices[d_slot_price][slot_id],
                length=slot_durations[d_slot_duration][slot_id]
            ))
            for slot_id in xrange(slot_qty)
        )
        
        # generate bidderInfo objects
        # 
        # campaign prices are defined as follows:
        # bidder_attrib_min = Σ(bidder_prio_values) * campaign_prio_sum / 100
        # campaign_budget = advert_price * ad_duration * bidder_attrib_min
        bidderInfos = {}
        for bidder_id in xrange(bidder_qty):
            bidder_prio_values = dict(enumerate(priority_bidders[bidder_id][d_inter_prio][d_prio_to_price][d_slot_price]))
            bidder_ad_duration = advert_durations[bidder_id][d_ad_duration]
            bidder_attrib_min = int(sum(bidder_prio_values.itervalues()) * campaign_min_prio_sum[bidder_id][d_min_prio] / 100)
            bidderInfos[bidder_id] = tvauction.processor.BidderInfo(
                id=bidder_id,
                budget=advert_prices[bidder_id][d_bid_price] * bidder_ad_duration * bidder_attrib_min,
                length=bidder_ad_duration,
                attrib_min=bidder_attrib_min,
                attrib_values=bidder_prio_values
            )
#            print bidderInfos[bidder_id].attrib_min, sum(bidderInfos[bidder_id].attrib_values.values())
        
        # calculate statistics about bidder demand
        slots_total_time = sum(s.length for s in slots.itervalues())
        bidder_info_stats = {}
        for b in bidderInfos.itervalues():
            min_count = 0
            max_count = 0
            # sort its priority vector ascending/descending to get min/max
            
            sort_by_prio = lambda a: a[1]
            count_prio = 0
            for slot_id,priority in sorted(b.attrib_values.iteritems(), key=sort_by_prio):
                if priority <= 0: continue
                count_prio += priority
                max_count += 1
                if count_prio >= b.attrib_min: break
            count_prio = 0
            for slot_id,priority in sorted(b.attrib_values.iteritems(), key=sort_by_prio,reverse=True):
                if priority <= 0: continue                
                count_prio += priority
                min_count += 1
                if count_prio >= b.attrib_min: break
            count_stats = (min_count,max_count,(max_count+min_count)/2)
            len_stats = tuple(b.length*v for v in count_stats)
            bidder_info_stats[b.id] = (count_stats,len_stats)
        
        print 'slots - amount: %d, total time: %d seconds' % (len(slots), slots_total_time)
        print 'bidders - amount: %d' % len(bidderInfos)
        print 'demand - min: %.2f %%, max: %.2f %%, avg: %.2f %%' % (
            100.0 * sum(v[1][0] for v in bidder_info_stats.itervalues()) / slots_total_time,
            100.0 * sum(v[1][1] for v in bidder_info_stats.itervalues()) / slots_total_time,
            100.0 * sum(v[1][2] for v in bidder_info_stats.itervalues()) / slots_total_time
        )
        
        print 'slot placements per bidder (min,max,avg):'
        for b_id,stat in bidder_info_stats.iteritems():
            print '  id %d: min %d, max %d, avg %d' % (b_id,stat[0][0],stat[0][1],stat[0][2])
        
        if not solve_problems:
            print 'not solving as requested'
            continue
        
        print 'solving...'
        calc_duration = -time.clock()
        res = auction_processor.solve(slots, bidderInfos, 60, 120, None)
        
        calc_duration += time.clock()
        print 'duration: %.1f seconds' % calc_duration
        
        print 'revenues:'
        for what in ('raw','vcg','core','final'): 
            print '  %s\t%d' % (what, sum(res['prices_%s' % what].itervalues()))
        
        if draw_results:
            graph_path = '/tmp/tvauction/%s-%s-%s' % (price_vector, core_algorithm, random_seed)
            if not os.path.exists(graph_path): os.makedirs(graph_path)
            now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            drawResult(graph_path+'/simulation_%s_%s' % (now,''.join(map(str,distribution))), res)
            
