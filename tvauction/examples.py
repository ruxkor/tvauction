from processor_pulp import Slot, BidderInfo, TvAuctionProcessor
import logging

import math

slot_amount = 168/4
bidder_amount = 50/2
bidder_flatten = bidder_amount+1


rand_increments = [175, 707, 312, 930, 276, 443, 468, 900, 855, 15, 658, 135, 238, 506, 244, 333, 912, 515, 458, 140, 925, 544, 720, 127, 545, 497, 962, 618, 900, 491, 515, 694, 738, 809, 75, 538, 422, 112, 106, 739, 1, 168, 554, 186, 762, 310, 888, 921, 164, 472, 538, 340, 267, 517, 412, 84, 941, 979, 713, 375, 501, 245, 149, 764, 74, 242, 385, 61, 910, 976, 775, 932, 661, 512, 757, 814, 443, 683, 795, 306, 955, 381, 202, 40, 908, 465, 755, 772, 125, 704, 934, 284, 712, 950, 645, 783, 525, 190, 587, 173]
rand_lengths = [60, 15, 45, 105, 90, 105, 15, 60, 30, 75, 90, 105, 105, 105, 60, 15, 75, 30, 60, 60, 30, 60, 30, 30, 90, 60, 90, 45, 45, 120, 105, 60, 60, 120, 90, 15, 45, 45, 45, 60, 75, 15, 30, 60, 60, 90, 120, 120, 60, 105]
rand_times = [4, 4, 10, 7, 7, 3, 5, 3, 9, 5, 9, 2, 7, 3, 5, 3, 10, 1, 2, 3, 7, 4, 3, 7, 8, 8, 5, 10, 5, 6, 10, 2, 6, 8, 4, 10, 4, 2, 8, 2, 9, 7, 1, 4, 5, 8, 1, 10, 3, 5]
rand_times = [i*3 for i in rand_times]

def example1():
    '''tests core pricing.'''
    slots = dict((i,Slot(i,1.0,120,1)) for i in range(3))
    bidderInfos = dict([
        (0,BidderInfo(0,1000,100,1,dict((i,1) for i in slots.iterkeys()))),
        (1,BidderInfo(1,1000,100,1,dict((i,1) for i in slots.iterkeys()))),
        (2,BidderInfo(2,1000,100,1,dict((i,1) for i in slots.iterkeys()))),
        (3,BidderInfo(3,1800,100,3,dict((i,1) for i in slots.iterkeys()))),
    ])
    return TvAuctionProcessor().solve(slots,bidderInfos)

def example2():
    '''tests selective attributes.'''
    slots = dict((i,Slot(i,1.0,120,1)) for i in range(3))
    bidderInfos = dict([
        (0,BidderInfo(0,1000,100,1,{0:0,1:0,2:1})),
        (1,BidderInfo(1,1000,100,1,{0:1,1:1,2:0})),
        (2,BidderInfo(2,1000,100,1,{0:1,1:0,2:0})),
        (3,BidderInfo(3,1800,100,3,{0:1,1:1,2:2})),
    ])
    return TvAuctionProcessor().solve(slots,bidderInfos)

def example3():
    '''tests for equal bids'''
    slot_amount = 168
    bidder_amount = 30
    slots = dict((i,Slot(i,0,120,1)) for i in range(slot_amount))
    bidderInfos = dict(
        (i,BidderInfo(i,1000,100,10,dict((i,1) for i in slots.iterkeys()))) 
        for i in range(bidder_amount)
    )
    return TvAuctionProcessor().solve(slots,bidderInfos)

def example4():
    '''tests for uncorrelated bids'''
    slot_amount = 168
    bidder_amount = 50
    slots = dict((i,Slot(i,0,120,1)) for i in range(slot_amount))
    bidderInfos = dict(
        (i,BidderInfo(i,incr*times*length,length,times,dict((i,1) for i in slots.iterkeys())))
        for (i,(incr,length,times)) 
        in enumerate(zip(rand_increments,rand_lengths,rand_times)[:bidder_amount])
    )
    return TvAuctionProcessor().solve(slots,bidderInfos)

def example5():
    '''tests for semicorrelated bids'''
    slot_amount = 168
    bidder_amount = 38
    slots = dict((i,Slot(i,0,120,1)) for i in range(slot_amount))
    bidderInfos = dict(
        (i,BidderInfo(i,(2*times*length)+incr,length,times,dict((i,1) for i in slots.iterkeys())))
        for (i,(incr,length,times)) 
        in enumerate(zip(rand_increments,rand_lengths,rand_times)[:bidder_amount])
    )
    proc = TvAuctionProcessor()
    return proc.solve(slots,bidderInfos)

def exampleRealistic1():
    slot_length = 120
    slot_amount = 336 # two weeks, each hour
    slot_baseline = 10000
    slot_price_baseline = 1.0 # the price for the baseline is 1.0 per second
    def slot_reach(slot_id):
        slot_hour = (slot_id) % 24
        return -0.2 * math.cos(slot_hour) # reach oscillates between +- 20% during the day
    slots = {}
    for slot_id in range(slot_amount):
        slot_modifier = slot_reach(slot_id)
        slot_modifier_sig = 1 if slot_modifier >= 0 else -1
        reach = int(slot_baseline*(1+slot_modifier))
        # the price growth/shrinks less than the reach, in our case as a sqrt
        price = round(slot_price_baseline * (1 + slot_modifier_sig*math.sqrt(abs(slot_modifier))),2)
        slots[slot_id] = Slot(slot_id,price,slot_length, reach)
        
    # each bidder wants 75000 as a baseline, which is approx. (sum(s.reach for s in slots)/ 50) * 1.1
    bidder_amount = 50
    bidder_reach_baseline = 150000
    bidder_reach_modifiers = [1.16, 1.45, 1.23, 1.85, 0.92, 1.29, 1.42, 1.53, 2.0, 1.89, 0.98, 1.94, 1.49, 1.68, 1.7, 1.11, 1.73, 1.76, 1.71, 0.92, 1.61, 1.54, 1.6, 0.81, 1.25, 1.05, 1.81, 1.92, 1.42, 1.36, 1.91, 1.59, 0.97, 0.88, 1.35, 1.6, 1.15, 1.94, 1.53, 1.99, 1.81, 1.44, 1.69, 1.07, 1.83, 1.46, 1.68, 1.8, 1.6, 1.73]
    bidder_price_baseline = 1.5
    bidder_price_modifiers = [1.18, 0.88, 1.04, 1.05, 1.06, 1.16, 1.09, 0.88, 1.15, 0.99, 1.15, 0.95, 1.02, 1.18, 1.14, 1.17, 1.16, 0.83, 1.19, 0.8, 0.87, 1.2, 0.85, 1.19, 0.86, 0.96, 1.03, 0.84, 0.91, 0.8, 1.14, 0.84, 1.1, 0.99, 1.2, 1.16, 1.18, 0.85, 1.13, 0.96, 1.08, 1.17, 0.92, 1.06, 1.1, 1.18, 0.81, 1.04, 1.0, 1.17]
    bidder_length_baseline = 30
    bidder_length_modifiers = [2.0, 1.59, 1.53, 1.58, 1.14, 1.0, 0.82, 1.72, 0.93, 1.93, 0.94, 0.83, 1.34, 1.91, 1.98, 1.45, 1.49, 0.83, 1.8, 1.32, 1.46, 1.03, 0.98, 1.75, 0.97, 0.9, 1.26, 1.74, 0.97, 1.43, 0.83, 1.8, 1.29, 1.57, 1.68, 1.93, 1.99, 1.4, 1.08, 1.61, 1.35, 1.93, 0.88, 1.54, 0.81, 1.55, 1.42, 1.27, 1.67, 1.12]

    # we use the predefined slot reach
    slot_reaches = dict((s_id,s.reach) for (s_id,s) in slots.iteritems())
    bidderInfos = {}
    for bidder_id,reach_mod,price_mod,length_mod in zip(range(bidder_amount),bidder_reach_modifiers,bidder_price_modifiers,bidder_length_modifiers):
        reach = int(bidder_reach_baseline*reach_mod)
        length = bidder_length_baseline * length_mod
        # price is correlated to reach and length
        budget = round(bidder_price_baseline * price_mod * length * reach,2)
        # ('id','budget','length','attrib_min','attrib_values'))
        bidderInfos[bidder_id] = BidderInfo(bidder_id, budget, length, reach, slot_reaches)
        
    proc = TvAuctionProcessor()
    return proc.solve(slots,bidderInfos)
        
def generate_random_color():
    import colorsys
    import random
    return colorsys.hls_to_rgb(
        random.random(), 
        float(random.randint(20,90))/100, 
        random.random()/2+0.5
    )

def drawit(save_path,res):
    import matplotlib.pyplot as plt
    import numpy as np
    ind = np.array(sorted(res['winners']))
    width = 0.8
    
    fig = plt.figure(None,figsize=(20,6))
    ax = fig.add_subplot(111)
    ax.grid(True,axis='y')
    
    bars = []
    bar_labels = []
    
    for (ptype,pcolor) in zip(['raw','vcg','core','final'],[(0,1,1),(0,0,0),(1,0,1),(1,1,0)]):
        vals = [v for (_k,v) in sorted(res['prices_%s' % ptype].iteritems())]
#        ax.bar(ind+(nr*width*1.8), vals, width, color=pcolor,alpha=0.3,linewidth=0)
        bar = ax.bar(ind, vals, width, color=pcolor,alpha=0.5,linewidth=0)
        bars.append(bar)
        bar_labels.append(ptype)

    ax.set_xticks(ind+0.4)
    ax.set_xticklabels(ind)
    ax.legend(bars,bar_labels)
    fig.savefig(save_path)
    
def main():    
    import json
    ex_fns = [example1,example2,example3,example4,example5][-1:]
    ex_names = ['example1','example2','example3','example4','example5'][-1:]
    for ex_n,ex_fn in zip(ex_names,ex_fns):
        res = ex_fn()
        drawit('/tmp/%s.pdf' % ex_n, res)
        print json.dumps(res)

def main_realistic():
    import json
    logging.basicConfig(level=logging.INFO)
    res = exampleRealistic1()
    drawit('/tmp/%s.pdf' % 'realistic', res)
    print json.dumps(res)
    
if __name__=='__main__':
    main()
