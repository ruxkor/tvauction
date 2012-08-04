from processor_pulp import Slot, BidderInfo, TvAuctionProcessor
import logging

slot_amount = 168/4
bidder_amount = 50/2
bidder_flatten = bidder_amount+1


rand_increments = [175, 707, 312, 930, 276, 443, 468, 900, 855, 15, 658, 135, 238, 506, 244, 333, 912, 515, 458, 140, 925, 544, 720, 127, 545, 497, 962, 618, 900, 491, 515, 694, 738, 809, 75, 538, 422, 112, 106, 739, 1, 168, 554, 186, 762, 310, 888, 921, 164, 472, 538, 340, 267, 517, 412, 84, 941, 979, 713, 375, 501, 245, 149, 764, 74, 242, 385, 61, 910, 976, 775, 932, 661, 512, 757, 814, 443, 683, 795, 306, 955, 381, 202, 40, 908, 465, 755, 772, 125, 704, 934, 284, 712, 950, 645, 783, 525, 190, 587, 173]
rand_lengths = [60, 15, 45, 105, 90, 105, 15, 60, 30, 75, 90, 105, 105, 105, 60, 15, 75, 30, 60, 60, 30, 60, 30, 30, 90, 60, 90, 45, 45, 120, 105, 60, 60, 120, 90, 15, 45, 45, 45, 60, 75, 15, 30, 60, 60, 90, 120, 120, 60, 105]
rand_times = [4, 4, 10, 7, 7, 3, 5, 3, 9, 5, 9, 2, 7, 3, 5, 3, 10, 1, 2, 3, 7, 4, 3, 7, 8, 8, 5, 10, 5, 6, 10, 2, 6, 8, 4, 10, 4, 2, 8, 2, 9, 7, 1, 4, 5, 8, 1, 10, 3, 5]
rand_times = [i*3 for i in rand_times]

def example1():
    '''tests core pricing.'''
    slots = dict((i,Slot(i,1.0,120)) for i in range(3))
    bidderInfos = dict([
        (0,BidderInfo(0,1000,100,1,dict((i,1) for i in slots.iterkeys()))),
        (1,BidderInfo(1,1000,100,1,dict((i,1) for i in slots.iterkeys()))),
        (2,BidderInfo(2,1000,100,1,dict((i,1) for i in slots.iterkeys()))),
        (3,BidderInfo(3,1800,100,3,dict((i,1) for i in slots.iterkeys()))),
    ])
    return TvAuctionProcessor().solve(slots,bidderInfos)

def example2():
    '''tests selective attributes.'''
    slots = dict((i,Slot(i,1.0,120)) for i in range(3))
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
    slots = dict((i,Slot(i,0,120)) for i in range(slot_amount))
    bidderInfos = dict(
        (i,BidderInfo(i,1000,100,10,dict((i,1) for i in slots.iterkeys()))) 
        for i in range(bidder_amount)
    )
    return TvAuctionProcessor().solve(slots,bidderInfos)

def example4():
    '''tests for uncorrelated bids'''
    slot_amount = 168
    bidder_amount = 50
    slots = dict((i,Slot(i,0,120)) for i in range(slot_amount))
    bidderInfos = dict(
        (i,BidderInfo(i,incr*times*length,length,times,dict((i,1) for i in slots.iterkeys())))
        for (i,(incr,length,times)) 
        in enumerate(zip(rand_increments,rand_lengths,rand_times)[:bidder_amount])
    )
    return TvAuctionProcessor().solve(slots,bidderInfos)

def example5():
    '''tests for semicorrelated bids'''
    slot_amount = 168
    bidder_amount = 50
    slots = dict((i,Slot(i,0,120)) for i in range(slot_amount))
    bidderInfos = dict(
        (i,BidderInfo(i,(2*times*length)+incr,length,times,dict((i,1) for i in slots.iterkeys())))
        for (i,(incr,length,times)) 
        in enumerate(zip(rand_increments,rand_lengths,rand_times)[:bidder_amount])
    )
    return TvAuctionProcessor().solve(slots,bidderInfos)

def generate_random_color():
    import colorsys
    import random
    return colorsys.hls_to_rgb(
        random.random(), 
        float(random.randint(20,90))/100, 
        random.random()/2+0.5
    )

def drawit(save_path,res):
#    res = {"winners": [1, 2, 35, 6, 33, 8, 10, 39, 15, 16, 17, 18, 20, 22, 23, 7, 25, 27, 30, 31], "prices_core": {"1": 0.0, "2": 3012.0, "35": 806.0, "6": 0.0, "33": 6569.0, "8": 1502.0, "10": 5518.0, "39": 992.0, "15": 0.0, "16": 5148.0, "17": 0.0, "18": 992.0, "20": 652.0, "22": 403.0, "23": 403.0, "7": 1581.0, "25": 3377.0, "27": 2902.0, "30": 6815.0, "31": 992.0}, "prices_vcg": {"1": 0.0, "2": 3012, "35": 806.0, "6": 0.0, "33": 6569, "8": 1502.0, "10": 5518, "39": 806.0, "15": 0.0, "16": 5148.0, "17": 0.0, "18": 652.0, "20": 652.0, "22": 42.0, "23": 42.0, "7": 1220.0, "25": 3377, "27": 2902.0, "30": 6815, "31": 806.0}, "prices_raw": {"1": 1067, "2": 3012, "35": 1438, "6": 918, "33": 6569, "8": 2475, "10": 5518, "39": 1459, "15": 603, "16": 5412, "17": 695, "18": 1178, "20": 2185, "22": 1260, "23": 1387, "7": 1980, "25": 3377, "27": 3318, "30": 6815, "31": 1414}}
    import matplotlib.pyplot as plt
    import numpy as np
    ind = np.array(sorted(res['winners']))
    width = 0.8
    
    fig = plt.figure(None,figsize=(20,6))
    ax = fig.add_subplot(111)
    ax.grid(True,axis='y')
    
    bars = []
    bar_labels = []
    
    for (ptype,pcolor) in zip(['raw','vcg','core'],[(0,1,1),(1,0,1),(1,1,0)]):
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
    logging.basicConfig(level=logging.INFO)
    ex_fns = [example1,example2,example3,example4,example5]
    ex_names = ['example1','example2','example3','example4','example5']
    for ex_n,ex_fn in zip(ex_names,ex_fns):
        res = ex_fn()
        drawit('/tmp/%s.pdf' % ex_n, res)
        print json.dumps(res)
        
if __name__=='__main__':
    main()
