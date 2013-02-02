# creates some nifty graphs for a scenario
import logging
from contextlib import closing
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib
import math

matplotlib.rc('font', family='serif')

extra_draw_info = {
    'bid': {'marker':'x', 'linestyle':':', 'markersize':5, 'color':'green'},
    'vcg': {'marker':'.', 'linestyle':'--', 'markersize':5, 'color':'blue'},
    'sep': {'marker':'.', 'linestyle':'-', 'markersize':5, 'color':'orange'},
    'ebpo': {'marker':'.', 'linestyle':'-.', 'markersize':5, 'color':'red'}
}

extra_draw_info['gwd'] = extra_draw_info['bid'].copy()
extra_draw_info['gwd']['markersize'] = 8

def _drawWinners(file_prefix, res, scenario):
    fig = plt.figure(None,figsize=(20,6))
    
    # the first graph shows all winners and how much they paid
    winners = np.array([k for (k,j) in res['winners']])
    width = 0.8
    
    ax1 = fig.add_subplot(111)
    ax1.grid(True, axis='y')
    
    bars = []
    bar_labels = []
    for nr, (ptype,pcolor) in enumerate(zip(['bid','vcg','core','final'],[(0,1,1),(0,0,0),(1,0,1),(1,1,0)])):
        bar_width = width - nr*width*0.2
        vals = [v for (k,v) in sorted(res['prices_%s' % ptype].iteritems()) if k in winners]
        bar = ax1.bar(winners-bar_width*0.5, vals, bar_width, color=pcolor,linewidth=0.5)
        bars.append(bar)
        bar_labels.append(ptype)

    ax1.set_xticks(winners)
    ax1.set_xticklabels(winners)
    ax_legend = ax1.legend(bars, bar_labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.12))
    fig.savefig(file_prefix+'_prices.svg', bbox_inches='tight', bbox_extra_artists=[ax_legend])
    
def _drawSteps(file_prefix, res, scenario):
    # the second graph is the step info graph    
    steps_info = res['step_info']
    fig = plt.figure(None,figsize=(10,3))
    ax2 = fig.add_subplot(111)
    ax2.grid(True, axis='y')
    
    step_max = 0
    tuples_all = {}
    for what in ('bid','vcg','sep','ebpo'):
        tuples_all[what] = tuples_what = [(nr, step_info[what]) for (nr, step_info) in enumerate(steps_info) if what in step_info]
        if tuples_what: step_max = max(step_max, *(s for (s,v) in tuples_what))
        
    for what in ('bid','vcg','sep','ebpo'):
        tuples_what = tuples_all[what]
        if not tuples_what: continue
        if tuples_what[-1][0] != step_max: tuples_what.append( (step_max,None) )
        
        steps_what = []
        values_what = []
        for nr, (step_what, value_what) in enumerate(tuples_what):
            if steps_what:
                step_prev, value_prev = tuples_what[nr-1]
                for i in range(step_prev+1,step_what):
                    steps_what.append(i)
                    values_what.append(value_prev)
            steps_what.append(step_what)
            values_what.append(value_what if value_what is not None else value_prev)
        ax2.plot(steps_what, values_what, drawstyle='steps-post',label=what, **extra_draw_info[what])
    ax2.set_ylabel('Value')
    ax2.set_xlabel('TRIM - Iteration')
    ax2.set_xlim(-0.1,step_max+0.1)
    ax_legend = ax2.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.09), prop={'size':10})
    fig.savefig(file_prefix+'_steps.svg', bbox_inches='tight', bbox_extra_artists=[ax_legend])
        
def _drawGaps(file_prefix, res, scenario):
    # gap graph
    fig = plt.figure(None,figsize=(10,3))
    ax3 = fig.add_subplot(111)
    ax3.grid(True, axis='y')
    
    gaps_by_type = {}
    # divide the gaps by class type
    for nr, (gap_type, gap) in enumerate(res['gaps']):
        gap_type = re.sub('_.*','',gap_type)
        if gap_type not in gaps_by_type: gaps_by_type[gap_type] = []
        gaps_by_type[gap_type].append((nr,gap))
     
    for gap_type, points in gaps_by_type.iteritems():
        points_x, points_y = zip(*points)
        ax3.plot(points_x, points_y, label=gap_type, linewidth=1.0, **extra_draw_info[gap_type])
        
    ax3.set_xlim(-0.5,len(res['gaps'])+0.5)
    ax3.set_ylabel('MIP Gap')
    ax3.set_xlabel('Iteration')
    ax_legend = ax3.legend(loc='upper center',ncol=len(gaps_by_type),bbox_to_anchor=(0.5,1.09), prop={'size':10})
    fig.savefig(file_prefix+'_gaps.svg', bbox_inches='tight', bbox_extra_artists=[ax_legend])
        
def _drawBidderInfos(file_prefix, res, scenario):
    # bidderinfo prices sparklines
    slots, bidder_infos = scenario
    fig = plt.figure(None,figsize=(16,9))
    fig_legends = []
    ax_slots = fig.add_subplot(len(bidder_infos)+1, 1, 1)
    points_x, points_y = zip(*( (s_id,s.price) for (s_id,s) in sorted(slots.iteritems())))
    ax_slots.plot(points_x, points_y, '-', drawstyle='steps-post', linewidth=0.5, color='grey', label='slots reserve price') 
    ax_slots.set_xticks([])
    ax_slots.set_yticks([])
    ax_slots.set_frame_on(False)
    fig_legends.append(ax_slots.legend(loc=(-0.1,0),frameon=False,prop={'size':6}))
    
    for nr, (k, bidder_info) in enumerate(sorted(bidder_infos.iteritems())):
        ax_bidder_attribs = fig.add_subplot(len(bidder_infos)+1, 1, nr+2)
        points_x, points_y = zip(*sorted(bidder_info.attrib_values.iteritems()))
        ax_bidder_attribs.plot(points_x, points_y, '-', drawstyle='steps-post', linewidth=0.5, label='bidder %d' % k)
        ax_bidder_attribs.set_xticks([])
        ax_bidder_attribs.set_yticks([])
        ax_bidder_attribs.set_frame_on(False)
        fig_legends.append(ax_bidder_attribs.legend(loc=(-0.1,0),frameon=False,prop={'size':6}))
    fig.savefig(file_prefix+'_bidder_attribs.svg', bbox_inches='tight', bbox_extra_artists=fig_legends)
    
def _drawSlotAssignments(file_prefix, res, scenario):
    slots, bidder_infos = scenario
    slots_y = dict((s_id, 0) for s_id in slots)
    
    fig = plt.figure(None,figsize=(100,10))
    ax = fig.add_subplot(111)
    
    bidders_data = []
    for k, s_assignments in sorted(res['winners_slots'].iteritems()):
        ad_length = bidder_infos[k].length
        bidder_assignment = sorted(s_assignments)
        bidder_height = [ad_length]*len(s_assignments)
        bidder_bottom = [s_y for (s_id,s_y) in sorted(slots_y.iteritems()) if s_id in bidder_assignment]
        bidders_data.append( (k, bidder_assignment, bidder_height, bidder_bottom) )
        # incremenet s_y height
        for s_id,s_y in slots_y.iteritems():
            if s_id in bidder_assignment:
                slots_y[s_id] += ad_length
    for nr, (k, bidder_assignment, bidder_height, bidder_bottom) in enumerate(bidders_data):
        bars = ax.bar(bidder_assignment, bidder_height, bottom=bidder_bottom, linewidth=0.5, edgecolor='grey', color=matplotlib.cm.jet(1.*nr/len(bidder_infos)))

#    bars = ax.bar(s_assignments, [ad_length]*len(s_assignments), bottom=sorted([s_y for (s_id,s_y) in slots_y.items() if s_id in s_assignments]))
    slots_remaining = sorted((s_id,s.length-slots_y[s_id]) for (s_id,s) in slots.iteritems())
    ax.bar(*zip(*slots_remaining), linewidth=0.5, color='white', edgecolor='grey', bottom=[s_y for (s_id,s_y) in sorted(slots_y.iteritems())])
    fig.savefig(file_prefix+'_slot_assignments.svg')

def drawResult(file_prefix, res, scenario):
    _drawWinners(file_prefix, res, scenario)
    _drawSteps(file_prefix, res, scenario)
    _drawGaps(file_prefix, res, scenario)
    _drawBidderInfos(file_prefix, res, scenario)
    _drawSlotAssignments(file_prefix, res, scenario)

if __name__ == '__main__':
    import sys
    import os
    import optparse
    import hashlib
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    from tvauction.common import json, convertToNamedTuples
        
    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    logging.basicConfig(level=log_level)
    
    parser = optparse.OptionParser()
    parser.set_usage('%prog [options] < result.pickle')
    parser.add_option('--scenopts', dest='scenopts', type='str', help='the options file used to generate the scenarios')
    parser.add_option('--scenarios', dest='scenarios', type='str', help='the scenarios file created by the generator')
    parser.add_option('--offset', dest='offset', type='int', default=None, help='the scenario offset')
    parser.add_option('--prefix', dest='add_prefix', type='str', help='anything else you would like to add to the graph filenames')
    parser.add_option('--graph-path', dest='graph_path', type='str', default='/tmp/tvauction_graphs', help='the base directory for the graphs. has to exist.')
    parser.epilog = 'If scenopts and offset is not passed, the first characters of the md5 of the scenario will be used'
    for option in parser.option_list: 
        if option.default != ("NO", "DEFAULT"): option.help += (" " if option.help else "") + "[default: %default]"
    if sys.stdin.isatty():
        print parser.format_help()
        sys.exit()
        
    options = parser.parse_args(sys.argv)[0]
    result = sys.stdin.read()
#    options.scenarios = '/tmp/scen.data'
#    options.scenopts = '/tmp/scen.opts'
#    result = open('/tmp/scen.result').read()
    
    if not os.path.exists(options.graph_path): 
        raise Exception('%s not existing' % options.graph_path)

    # get sha1
    m = hashlib.md5(result)
    graph_file_prefix = m.hexdigest()[:10]
    
    if options.add_prefix: graph_file_prefix += options.add_prefix
        
    # decode the whole result as object
    result = json.decode(result)
    
    # prefix it with infos about the scenario if available
    if options.scenopts and options.offset is not None:
        with closing(open(options.scenopts,'r')) as scenopts_file:
            generated_options = json.decode(scenopts_file.read())
#            the generator loops first through all random seeds, and then through all distributions
            used_random_seed = generated_options['random_seeds'][
                int(math.ceil(options.offset / len(generated_options['distributions'])))
            ]
            used_distribution = generated_options['distributions'][options.offset - generated_options['random_seeds'].index(used_random_seed)]
            graph_file_scen_prefix = '%s_%s' % ('-'.join(map(str,used_distribution)),used_random_seed)
            graph_file_prefix = '%s_%s' % (graph_file_scen_prefix,graph_file_prefix)
    
    scenario = None
    if options.scenarios and options.offset is not None:
        with closing(open(options.scenarios,'r')) as scenario_file:
            scenarios = json.decode(scenario_file.read())
            scenario = scenarios[options.offset]
            convertToNamedTuples(scenario)
    drawResult(options.graph_path+'/run_%s' % graph_file_prefix, result, scenario)
    
