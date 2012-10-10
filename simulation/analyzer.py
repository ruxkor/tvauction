# creates analyses about the campaign situation
import logging

def analyze(scenario):
    slots, bidder_infos = scenario
    slots_total_time = sum(s.length for s in slots.itervalues())
    bidder_info_stats = {}
    for b in bidder_infos.itervalues():
        min_count = 0
        max_count = 0
        # sort its priority vector ascending/descending to get min/max
        
        total_prio = sum(b.attrib_values.itervalues())
        sort_by_prio = lambda a: a[1]
        for slot_id,priority in sorted(b.attrib_values.iteritems(), key=sort_by_prio):
            if priority <= 0: continue
            total_prio -= priority
            max_count += 1
            if total_prio < b.attrib_min: break
        total_prio = sum(b.attrib_values.itervalues())
        for slot_id,priority in sorted(b.attrib_values.iteritems(), key=sort_by_prio,reverse=True):
            if priority <= 0: continue                
            total_prio -= priority
            min_count += 1
            if total_prio < b.attrib_min: break
        count_stats = (min_count,max_count,(max_count+min_count)/2)
        len_stats = tuple(b.length*v for v in count_stats)
        bidder_info_stats[b.id] = (count_stats,len_stats)
    
    res = {
        'slots': {
            'quantity': len(slots),
            'total_time': slots_total_time
        },
        'bidders': {
            'quantity': len(bidder_infos)
        },
        'demand': {
            'min': round(float(sum(v[1][0] for v in bidder_info_stats.itervalues())) / slots_total_time, 2),
            'max': round(float(sum(v[1][1] for v in bidder_info_stats.itervalues())) / slots_total_time, 2),
            'avg': round(float(sum(v[1][2] for v in bidder_info_stats.itervalues())) / slots_total_time, 2)
        },
        'appearances': dict(
            (b_id, {'min': stat[0][0], 'max': stat[0][1], 'avg': stat[0][2]})
            for b_id,stat in bidder_info_stats.iteritems()
        )
    }
    return res

if __name__=='__main__':
    import sys
    import os
    import optparse
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    from tvauction.common import json, convertToNamedTuples
        
    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    logging.basicConfig(level=log_level)
    
    parser = optparse.OptionParser()
    parser.set_usage('%prog [options] < scenarios.json')
    if sys.stdin.isatty():
        print parser.format_help()
        sys.exit()
        
    for option in parser.option_list: 
        if option.default != ("NO", "DEFAULT"): option.help += (" " if option.help else "") + "[default: %default]"
    options = parser.parse_args(sys.argv)[0]
    scenarios = json.decode(sys.stdin.read())
    # convert
    for scenario in scenarios:
        convertToNamedTuples(scenario)
        res = analyze(scenario)
        print json.encode(res)

    