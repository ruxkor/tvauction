import unittest
import logging
import sys
sys.path.insert(0,'../tvauction')
from common import Slot, BidderInfo
from processor import TvAuctionProcessor

class Test(unittest.TestCase):

    def testName(self):
        slots = dict((i,Slot(i,1.0,120)) for i in range(3))
        multiple = ((2000,3), (800,1))
        bidderInfos = dict([
            (0,BidderInfo(0,100,((1000,1),),dict((i,1) for i in slots.iterkeys()))),
            (1,BidderInfo(1,100,((1000,1),),dict((i,1) for i in slots.iterkeys()))),
            (2,BidderInfo(2,100,((1000,1),),dict((i,1) for i in slots.iterkeys()))),
            (3,BidderInfo(3,100,((1800,3),),dict((i,1) for i in slots.iterkeys()))),
            (4,BidderInfo(4,100,multiple,dict((i,1) for i in slots.iterkeys()))),
        ])
        
        processor = TvAuctionProcessor()
        processor.core_algorithm = processor.coreClass.TRIM_VALUES
        processor.solve(slots, bidderInfos)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    logging.basicConfig(level=logging.INFO)
    unittest.main()