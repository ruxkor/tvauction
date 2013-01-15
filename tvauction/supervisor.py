#!/usr/bin/env python
# -*- coding: utf-8; -*-

import logging
from common import json, Slot, BidderInfo, convertToNamedTuples
from processor import TvAuctionProcessor as proc

import gevent
import gevent_zeromq as zmq
from gevent.queue import Queue
from multiprocessing import Process

def serialize(data):
    return json.encode(data)

def unserialize(data):
    return json.decode(data)
    
class ProcessorTask(Process):
    def __init__(self):
        Process.__init__(self)
        self.ctx = zmq.Context()
        self.socket_supervisor = self.ctx.socket(zmq.REP)
        self.socket_supervisor.connect('ipc://processor_task')
    def run(self):
        print 'worker'
        params = unserialize(self.socket_supervisor.recv())
        try:
            scenario, options = params
            convertToNamedTuples(scenario)
            slots, bidder_infos = scenario
            res = proc.solve(slots, bidder_infos, **options)
            self.socket_supervisor.send((None, res))
        except Exception, err:
            self.socket_supervisor.send([err, None])


class Supervisor(object):
    def __init__(self, ctx, config):
        self.socket_middleware_sub = ctx.socket(zmq.SUB)
        self.socket_middleware_sub.connect(config.uri_middleware_sub)
        self.socket_middleware_rr = ctx.socket(zmq.REQ) 
        self.socket_middleware_rr.connect(config.uri_middleware_rr)
        self.socket_worker = ctx.socket(zmq.DEALER)
        self.socket_worker.bind('ipc://processor_task')
        self.queue = Queue()
        self.worker = None
    
    def initialize(self):
        return (
            gevent.spawn(self.handleMiddlewareSubscription),
            gevent.spawn(self.handleMiddlewareReqRep),
            gevent.spawn(self.handleWorker),
        )
    
    def isFree(self):
        return not(self.worker and self.worker.is_alive())
    
    def handleMiddlewareSubscription(self):
        print 'handler1'
        while True:
            data = unserialize(self.socket_middleware_sub.recv())
            logging.debug('--middleware--> %s' % data)
            action, params = data
            if action == 'is_free':
                resp = [action, self.isFree()]
                self.queue.put(resp)
            elif action == 'solve' and self.isFree():
                if self.worker: self.worker.terminate()
                self.worker = ProcessorTask()
                self.worker.start()
                self.socket_worker.send("", zmq.SNDMORE)
                self.socket_worker.send(serialize(params))
            else:
                resp = [action, 'error']
                self.queue.put(resp)
    
    def handleMiddlewareReqRep(self):
        print 'handler2'
        while True:
            data = self.queue.get()
            self.socket_middleware_rr.send(serialize(data))
            resp = unserialize(self.socket_middleware_rr.recv())
            logging.info('response: %s', resp)
    
    def handleWorker(self):
        print 'handler3'
        while True:
            self.socket_worker.recv()
            data = Supervisor.unserialize(self.socket_worker.recv())
            self.queue.put(['result',data])
    
def main():
    import os
    import sys
    from optparse import OptionParser

    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    logging.basicConfig(level=log_level)
    
    parser = OptionParser()
#    parser.set_usage('%prog [options] < scenarios.json')
    parser.add_option('--uri_middleware-sub', dest='uri_middleware_sub', default='tcp://127.0.0.1:10200')
    parser.add_option('--uri_middleware-rr', dest='uri_middleware_rr', default='tcp://127.0.0.1:10200')
    options = parser.parse_args()[0]
    
    logging.info('started')
    ctx = zmq.Context()
    supervisor = Supervisor(ctx, options)
    gevent.joinall(supervisor.initialize())
if __name__ == '__main__':
    main()
