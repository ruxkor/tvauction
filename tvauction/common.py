import simplejson
from collections import namedtuple

Slot = namedtuple('Slot', ('id','price','length'))
BidderInfo = namedtuple('BidderInfo', ('id','length','bids','attrib_values'))
# bids: [(price, amount),...]
# attrib_values {slot_id_1: value_1, ...}

class _JSONEncoder(simplejson.JSONEncoder):
    def default(self, o):
        try: iterable = iter(o)
        except TypeError: pass
        else: return list(iterable)
        return simplejson.JSONEncoder.default(self, o)
class _JSON(object):
    '''JSON implementation used for en/decoding'''
    def __init__(self):
        self.encoder = _JSONEncoder(ensure_ascii=False,separators=(',', ':'))
        self.decoder = simplejson.JSONDecoder(object_hook=self.transformNumericKeys)
        
    @classmethod
    def transformTuplesToStrings(cls, obj):
        for k,v in obj.items():
            if type(k) in (set,frozenset):
                obj[str(k)] = v
                del obj[k]
    
    @classmethod
    def transformStringsToTuples(cls, obj):
        for k,v in obj.items():
            if k[0] == '(' and k[-1] == ')' and ',' in k:
                k_new = tuple(int(v) for v in k[1:-1].split(',') if len(v))
                obj[k_new] = v
                del obj[k]
                
    @classmethod
    def transformNumericKeys(cls, obj):
        for k,v in obj.items():
            try:
                k_int = int(k)
                del obj[k]
                obj[k_int] = v
            except ValueError: pass
        return obj
    
    @classmethod
    def transformBack(cls, obj):
        cls.transformNumericKeys(obj)
        cls.transformStringsToTuples(obj)
    
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

