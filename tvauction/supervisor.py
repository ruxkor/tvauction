#!/usr/bin/env python
# -*- coding: utf-8; -*-

import logging
from common import json, convertToNamedTuples
from processor import TvAuctionProcessor

import gevent
from gevent.queue import Queue
from gevent.event import Event

import gevent_zeromq as gzmq
from multiprocessing import Process

import zmq as tzmq
import uuid
def serialize(data):
    return json.encode(data)

def unserialize(data):
    return json.decode(data)
    
class SupervisorTask(Process):
    def __init__(self):
        Process.__init__(self)
        
    def run(self):
        ctx = tzmq.Context()
        socket_supervisor = ctx.socket(tzmq.REP)
        socket_supervisor.connect('ipc://supervisor_task.ipc')
        data = unserialize(socket_supervisor.recv())
        try:
            auction_id, scenario, options = data
            convertToNamedTuples(scenario)
            res = self._solve(scenario, options)
            socket_supervisor.send(serialize([None, auction_id, res]))
        except Exception, err:
            socket_supervisor.send(serialize(['%s' % err, auction_id, None]))
            
    def _solve(self, scenario, options):
        slots, bidder_infos = scenario
        proc = TvAuctionProcessor()
        res = proc.solve(slots, bidder_infos, **options)
        return res        


class Supervisor(object):
    worker_class = SupervisorTask
    
    def __init__(self, ctx, uri_middleware_sub, uri_middleware_rr):
        self.id = uuid.uuid4()
        self.socket_middleware_sub = ctx.socket(gzmq.SUB)
        self.socket_middleware_rr = ctx.socket(gzmq.REQ) 
        self.socket_worker = ctx.socket(gzmq.DEALER)
        self.socket_middleware_sub.connect(uri_middleware_sub)
        self.socket_middleware_rr.connect(uri_middleware_rr)
        self.socket_worker.bind('ipc://supervisor_task.ipc')
        self.socket_middleware_sub.setsockopt(gzmq.SUBSCRIBE,'')
        self.queue = Queue()
        self.worker = None
        self.event_shut_down = None
    
    def initialize(self):
        return (
            gevent.spawn(self.handleMiddlewareSubscription),
            gevent.spawn(self.handleMiddlewareReqRep),
            gevent.spawn(self.handleWorker),
        )
    
    def isFree(self):
        return not(self.worker and self.worker.is_alive())
    
    def shutdown(self):
        if self.event_shut_down:
            raise Exception('already shutting down')
        self.event_shut_down = Event()
        if self.worker and self.worker.is_alive():
            self.worker.terminate()
        self.socket_middleware_rr.close()
        self.socket_middleware_sub.close()
        self.socket_worker.close()
        
    def handleMiddlewareSubscription(self):
        while True:
            data = unserialize(self.socket_middleware_sub.recv())
            logging.debug('handleMiddlewareSubscription\tgot: %s', data)
            action, _params = data[0], data[1:]
            if action == 'is_free':
                resp = [action, None, self.isFree()]
                self.queue.put(resp)
            else:
                resp = [action, 'unknown action', None]
                self.queue.put(resp)
    
    def handleMiddlewareReqRep(self):
        while True:
            data = self.queue.get()
            
            logging.debug('handleMiddlewareReqRep\tgot: %s', data)
            self.socket_middleware_rr.send(serialize(data))

            resp = self.socket_middleware_rr.recv()
            if not resp: continue
            
            resp = unserialize(resp)
            action, auction_id, params = resp[0], resp[1], resp[2:] 
            
            logging.info('response: %s', resp)
            if action == 'solve' and self.isFree():
                if self.worker: self.worker.terminate()
                self.worker = self.worker_class()
                self.worker.start()
                self.socket_worker.send('', gzmq.SNDMORE)
                self.socket_worker.send(serialize([auction_id]+params))
            elif action == 'solve':
                self.queue.put(['solve', 'in_use', auction_id, None])
                logging.error('got response but am not free')
    
    def handleWorker(self):
        while True:
            self.socket_worker.recv()
            data = unserialize(self.socket_worker.recv())
            logging.debug('handleWorker\tgot: %s', data)
            self.queue.put(['solve'] + data)
            
    
def main():
    import os
    from optparse import OptionParser

    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    logging.basicConfig(level=log_level)
    
    parser = OptionParser()
#    parser.set_usage('%prog [options] < scenarios.json')
    parser.add_option('--uri_middleware-sub', dest='uri_middleware_sub', default='tcp://127.0.0.1:10234')
    parser.add_option('--uri_middleware-rr', dest='uri_middleware_rr', default='tcp://127.0.0.1:10235')
    options = parser.parse_args()[0]
    
    logging.info('started')
    ctx = gzmq.Context()
    supervisor = Supervisor(ctx, options.uri_middleware_sub, options.uri_middleware_rr)
    gevent.joinall(supervisor.initialize())


if __name__ == '__main__':
    main()
