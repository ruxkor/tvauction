from processor_pulp import Slot, BidderInfo, solve as processor_solve
import logging

slot_amount = 168/4
bidder_amount = 50/2
bidder_flatten = bidder_amount+1


rand_increments = [175, 707, 312, 930, 276, 443, 468, 900, 855, 15, 658, 135, 238, 506, 244, 333, 912, 515, 458, 140, 925, 544, 720, 127, 545, 497, 962, 618, 900, 491, 515, 694, 738, 809, 75, 538, 422, 112, 106, 739, 1, 168, 554, 186, 762, 310, 888, 921, 164, 472, 538, 340, 267, 517, 412, 84, 941, 979, 713, 375, 501, 245, 149, 764, 74, 242, 385, 61, 910, 976, 775, 932, 661, 512, 757, 814, 443, 683, 795, 306, 955, 381, 202, 40, 908, 465, 755, 772, 125, 704, 934, 284, 712, 950, 645, 783, 525, 190, 587, 173]
rand_lengths = [60, 15, 45, 105, 90, 105, 15, 60, 30, 75, 90, 105, 105, 105, 60, 15, 75, 30, 60, 60, 30, 60, 30, 30, 90, 60, 90, 45, 45, 120, 105, 60, 60, 120, 90, 15, 45, 45, 45, 60,
 75, 15, 30, 60, 60, 90, 120, 120, 60, 105]
rand_times = [4, 4, 10, 7, 7, 3, 5, 3, 9, 5, 9, 2, 7, 3, 5, 3, 10, 1, 2, 3, 7, 4, 3, 7, 8, 8, 5, 10, 5, 6, 10, 2, 6, 8, 4, 10, 4, 2, 8, 2, 9, 7, 1, 4, 5, 8, 1, 10, 3, 5]
rand_times = [i*3 for i in rand_times]

def example1():
    '''tests core pricing.'''
    slots = [Slot(i,1.0,120) for i in range(3)]
    bidderInfos = [
        BidderInfo(0,1000,100,1,(1,)*len(slots)),
        BidderInfo(1,1000,100,1,(1,)*len(slots)),
        BidderInfo(2,1000,100,1,(1,)*len(slots)),
        BidderInfo(3,1800,100,3,(1,)*len(slots)),
    ]
    return processor_solve(slots,bidderInfos)

def example2():
    '''tests selective attributes.'''
    slots = [Slot(i,1.0,120) for i in range(3)]
    bidderInfos = [
        BidderInfo(0,1000,100,1,(0,0,1)),
        BidderInfo(1,1000,100,1,(1,1,0)),
        BidderInfo(2,1000,100,1,(1,0,0)),
        BidderInfo(3,1800,100,3,(1,1,2)),
    ]
    return processor_solve(slots,bidderInfos)

def example3():
    '''tests for equal bids'''
    slot_amount = 168
    bidder_amount = 30
    slots = [Slot(i,0,120) for i in range(slot_amount)]
    bidderInfos = [BidderInfo(i,1000,100,10,(1,)*len(slots)) for i in range(bidder_amount)]
    return processor_solve(slots,bidderInfos)

def example4():
    '''tests for uncorrelated bids'''
    slot_amount = 168
    bidder_amount = 30
    slots = [Slot(i,0,120) for i in range(slot_amount)]
    bidderInfos = [
        BidderInfo(i,incr*times*length,length,times,(1,)*len(slots)) 
        for (i,(incr,length,times)) 
        in enumerate(zip(rand_increments,rand_lengths,rand_times))
    ][:bidder_amount]
    return processor_solve(slots,bidderInfos)

def example5():
    '''tests for semicorrelated bids'''
    slot_amount = 168
    bidder_amount = 40
    slots = [Slot(i,0,120) for i in range(slot_amount)]
    bidderInfos = [
        BidderInfo(i,(2*times*length)+incr,length,times,(1,)*len(slots)) 
        for (i,(incr,length,times)) 
        in enumerate(zip(rand_increments,rand_lengths,rand_times))
    ][:bidder_amount]
    return processor_solve(slots,bidderInfos)

if __name__=='__main__':
    import json
    from pprint import pprint as pp
    logging.basicConfig(level=logging.INFO)
#    print json.dumps(example1())
#    print json.dumps(example2())
#    print json.dumps(example3())
#    print json.dumps(example4())
    print json.dumps(example5())
