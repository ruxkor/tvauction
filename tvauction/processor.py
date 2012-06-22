import numpy as np
from openopt import MILP
from FuncDesigner import *
from collections import namedtuple,defaultdict
from FuncDesigner.ooVar import oovar
import json
import pprint

Slot = namedtuple('Slot', ('price','length'))
BidderInfo = namedtuple('BidderInfo', ('budget','length','attrib_min','attribs'))

def xi_factory(pref):
    vals = {'i':0}
    def xi_gen():
        vals['i'] += 1
        return '%s_%d' % (pref,vals['i'])
        return oovar('%s_%d' % (pref,vals['i']), domain=bool)
    return xi_gen

def x_factory(pref):
    vals = {'i':0}
    def x_gen():
        vals['i'] += 1
        return defaultdict(xi_factory('%s_%d' % (pref,vals['i'])))
    return x_gen

def wdp(slots,bidderInfos):
    '''the winner determination, implemented as a multiple knapsack problem'''
    slots_len = len(slots)
    bidderInfos_len = len(bidderInfos)
    x = defaultdict(dict)
    for i in range(slots_len):
        for j in range(bidderInfos_len):
            x[i][j] = oovar('x_%d_%d' % (i,j), domain=bool)
    
    y = {}
    for j in range(bidderInfos_len):
        y[j] = oovar('y_%d' % (j,), domain=bool)
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
    
    
    print 'building'
    p = MILP(obj, startPoint, constraints=cons)
    print 'solving'
    r = p.maximize('glpk', iprint=-1)
    
    
    pprint.pprint(r.xf)
    print r.ff

slots = [Slot(1.0,120) for i in range(100)]

bidderInfos = [BidderInfo(210,100,2,[]) for i in range(20)]
wdp(slots, bidderInfos)
