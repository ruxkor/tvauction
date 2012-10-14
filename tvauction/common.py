import simplejson
from collections import namedtuple

Slot = namedtuple('Slot', ('id','price','length'))
BidderInfo = namedtuple('BidderInfo', ('id','budget','length','attrib_min','attrib_values'))

class _JSON(object):
    '''JSON implementation used for en/decoding'''
    def __init__(self):
        self.encoder = simplejson.JSONEncoder(ensure_ascii=False,separators=(',', ':'))
        self.decoder = simplejson.JSONDecoder(object_hook=self.transformNumericKeys)

    @staticmethod
    def transformNumericKeys(obj):
        for k,v in obj.iteritems():
            try:
                k_int = int(k)
                del obj[k]
                obj[k_int] = v
            except ValueError: pass
        return obj
    
    def encode(self,obj):
        '''encode an object, returning a utf8 encoded string (not a unicode string!)'''
        return self.encoder.encode(obj).encode('utf-8')
    
    def decode(self,string):
        return self.decoder.decode(string)
json = _JSON()

def convertToNamedTuples(scenario):
    '''converts the passed scenario IN-PLACE into the respective namedtuples'''
    scenario_slots, scenario_bidders = scenario
    for slot_id,slot_tuple in scenario_slots.iteritems():
        scenario_slots[slot_id] = Slot(**slot_tuple) 
    for bidder_id,bidder_tuple in scenario_bidders.iteritems():
        scenario_bidders[bidder_id] = BidderInfo(**bidder_tuple)

