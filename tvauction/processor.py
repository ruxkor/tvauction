from openopt import MILP
from FuncDesigner import *
from collections import namedtuple,defaultdict
from FuncDesigner.ooVar import oovar
import pprint
from datetime import datetime
import random
Slot = namedtuple('Slot', ('price','length'))
BidderInfo = namedtuple('BidderInfo', ('budget','length','attrib_min','attribs'))

time_start = datetime.now()

def wdp(slots,bidderInfos):
    '''the winner determination, implemented as a multiple knapsack problem'''
    slots_len = len(slots)
    bidderInfos_len = len(bidderInfos)
    x = defaultdict(dict)
    for i in range(slots_len):
        for j in range(bidderInfos_len):
            x[i][j] = oovar('x_%d_%d' % (i,j), domain=bool, size=1)
    
    y = {}
    for j in range(bidderInfos_len):
        y[j] = oovar('y_%d' % (j,), domain=bool, size=1)
    cons = []
        
#    the sum of all assigned ad length has to be at most the length of the slot
    for (i,slot) in enumerate(slots):
        f = sum(bidderInfo.length*x[i][j] for (j,bidderInfo) in enumerate(bidderInfos))
        cons.append(f <= slot.length)
        
#    match the bidders' demands regarding their atttributes
    for (j,bidderInfo) in enumerate(bidderInfos):
        M = slots_len+2
        f = sum(x[i][j] for i in range(slots_len))
        f2 = bidderInfo.attrib_min - f
        cons.append(f <= M*y[j])
        cons.append(f2 <= M*(1-y[j]))
    
#    user can at most spend the maximum price
    for (j,bidderInfo) in enumerate(bidderInfos):
        f = sum(bidderInfo.length*slot.price*x[i][j] for (i,slot) in enumerate(slots))
        cons.append(f <= bidderInfo.budget)
    
#    oovar domain=bool takes already care of min and max bounds

    # set dimensions of oovars
    startPoint = {}
    for j in range(bidderInfos_len):
        startPoint[y[j]] = 0
        for i in range(slots_len):
            startPoint[x[i][j]] = 0
    
    obj = sum(bidderInfo.budget*y[j] for (j,bidderInfo) in enumerate(bidderInfos))
    ooPoint
    
    print 'building', datetime.now()-time_start
    p = MILP(obj, startPoint, constraints=cons)
    print 'solving', datetime.now()-time_start
    r = p.maximize('lpSolve',iprint=0)
    
    return r

slot_amount = 168
bidder_amount = 40

slots = [Slot(1.0,120) for i in range(slot_amount)]
rand_increments = [175, 707, 312, 930, 276, 443, 468, 900, 855, 15, 658, 135, 238, 506, 244, 333, 912, 515, 458, 140, 925, 544, 720, 127, 545, 497, 962, 618, 900, 491, 515, 694, 738, 809, 75, 538, 422, 112, 106, 739, 1, 168, 554, 186, 762, 310, 888, 921, 164, 472, 538, 340, 267, 517, 412, 84, 941, 979, 713, 375, 501, 245, 149, 764, 74, 242, 385, 61, 910, 976, 775, 932, 661, 512, 757, 814, 443, 683, 795, 306, 955, 381, 202, 40, 908, 465, 755, 772, 125, 704, 934, 284, 712, 950, 645, 783, 525, 190, 587, 173]
bidderInfos = [BidderInfo(1000+i,100,10,[]) for i in rand_increments[:bidder_amount]]

r = wdp(slots, bidderInfos)
print 'time', datetime.now()-time_start
print r.ff, 'objective value'

#pprint.pprint([(var,val) for (var,val) in r.xf.iteritems() if val])
