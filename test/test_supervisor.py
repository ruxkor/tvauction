import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../tvauction')

import unittest
import logging

import gevent
import gevent_zeromq as zmq

from common import json,Slot,BidderInfo
from supervisor import Supervisor, unserialize, serialize

ctx = zmq.Context()

config = {
    'uri_pub':'ipc://test_uri_pub.ipc',
    'uri_rr':'ipc://test_uri_rr.ipc',
}


class Test(unittest.TestCase):
    def generateScenario(self):
        slots = dict((i,Slot(i,1.0,120)) for i in range(3))
        bidderInfos = dict([
            (0,BidderInfo(0,1000,100,1,{0:0,1:0,2:1})),
            (1,BidderInfo(1,1000,100,1,{0:1,1:1,2:0})),
            (2,BidderInfo(2,1000,100,1,{0:1,1:0,2:0})),
            (3,BidderInfo(3,1800,100,3,{0:1,1:1,2:2})),
        ])
        return slots, bidderInfos
        
    def setUp(self):
        self.socket_pub = ctx.socket(zmq.PUB)
        self.socket_rr = ctx.socket(zmq.REP)
        self.socket_pub.bind(config['uri_pub'])
        self.socket_rr.bind(config['uri_rr'])
        self.supervisor = Supervisor(ctx, config['uri_pub'], config['uri_rr'])
        self.supervisor_handlers = self.supervisor.initialize()
        gevent.sleep(0.01)
        
    def tearDown(self):
        gevent.killall(self.supervisor_handlers)
        self.supervisor.shutdown()
        self.socket_rr.close()
        self.socket_pub.close()
        
    def testIsFree(self):
        def send():
            self.socket_pub.send(serialize(["is_free"]))
        def handle():
            data = json.decode(self.socket_rr.recv())
            self.assertEquals(['is_free', True], data)
            self.socket_rr.send('{"ui":2}')
        gevent.joinall((
            gevent.spawn(handle), 
            gevent.spawn(send)
        ), timeout=1.0,  raise_error=True)
        
    def testIsNotFree(self):
        self.supervisor.isFree = lambda *a: False
        def send():
            self.socket_pub.send(serialize(["is_free"]))
        def handle():
            data = json.decode(self.socket_rr.recv())
            self.assertEquals(['is_free', False], data)
            self.socket_rr.send('')
        gevent.joinall((
            gevent.spawn(handle), 
            gevent.spawn(send)
        ), timeout=1.0,  raise_error=True)
        
    def testSendEvenIfNotFree(self):
        self.supervisor.isFree = lambda *a: False
        def send():
            self.socket_pub.send(serialize(["is_free"]))
        def handle():
            data = json.decode(self.socket_rr.recv())
            self.assertEquals(['is_free', False], data)
            self.socket_rr.send('{"ui":2}')
            data = json.decode(self.socket_rr.recv())
            self.assertEquals(['in_use', None], data)
        gevent.joinall((
            gevent.spawn(handle), 
            gevent.spawn(send)
        ), timeout=1.0,  raise_error=True)
        
    def testSendInvalidData(self):
        def send():
            self.socket_pub.send(serialize(["is_free"]))
        def handle():
            self.socket_rr.recv()
            self.assertTrue(self.supervisor.isFree())
            self.socket_rr.send('["invalid"]')
            data = unserialize(self.socket_rr.recv())
            self.assertIsNotNone(data[0])
        gevent.joinall((
            gevent.spawn(handle), 
            gevent.spawn(send)
        ), timeout=1.0,  raise_error=True)

    def testSendValidData(self):
        def send():
            self.socket_pub.send(serialize(["is_free"]))
        def handle():
            self.socket_rr.recv()
            self.assertTrue(self.supervisor.isFree())
            scenario_data = [self.generateScenario(), {}]
            self.socket_rr.send(serialize(scenario_data))
            data = unserialize(self.socket_rr.recv())
            self.assertIsNone(data[0])
        gevent.joinall((
            gevent.spawn(handle), 
            gevent.spawn(send)
        ), timeout=1.0,  raise_error=True)    
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    unittest.main()