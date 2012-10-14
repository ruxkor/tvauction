# creates some nifty graphs for a scenario
import logging
from contextlib import closing
import matplotlib.pyplot as plt
import numpy as np
import re

def drawResult(file_prefix, res):
    fig = plt.figure(None,figsize=(20,6))
    
    # the first graph shows all winners and how much they paid
    ind = np.array(sorted(res['winners']))
    width = 0.8
    
    ax1 = fig.add_subplot(111)
    ax1.grid(True, axis='y')
    
    bars = []
    bar_labels = []
    
    for (nr,(ptype,pcolor)) in enumerate(zip(['bid','vcg','core','final'],[(0,1,1),(0,0,0),(1,0,1),(1,1,0)])):
        bar_width = width - nr*width*0.2
        vals = [v for (k,v) in sorted(res['prices_%s' % ptype].iteritems()) if k in ind]
        bar = ax1.bar(ind-bar_width*0.5, vals, bar_width, color=pcolor,linewidth=0.5)
        bars.append(bar)
        bar_labels.append(ptype)

    ax1.set_xticks(ind)
    ax1.set_xticklabels(ind)
    ax1.legend(bars, bar_labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.12))
    fig.savefig(file_prefix+'_prices.pdf')

    # the second graph is the step info graph    
    steps_info = res['step_info']
    fig = plt.figure(None,figsize=(16,9))
    ax2 = fig.add_subplot(111)
    ax2.grid(True, axis='y')
    
    step_max = 0
    tuples_all = {}
    for what in ('bid','vcg','sep','blocking_coalition','ebpo'):
        tuples_all[what] = tuples_what = [(nr, step_info[what]) for (nr, step_info) in enumerate(steps_info) if what in step_info]
        if tuples_what: step_max = max(step_max, *(s for (s,v) in tuples_what))
        
    for what in ('bid','vcg','sep','blocking_coalition','ebpo'):
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
        ax2.plot(steps_what, values_what, '.', drawstyle='steps-post',label=what,linestyle='-', linewidth=2.0, markersize=10.0)
    
    
    ax2.set_xlim(-0.1,step_max+0.1)
    ax2.legend(loc='upper center',ncol=5,bbox_to_anchor=(0.5,1.09))
    fig.savefig(file_prefix+'_steps.pdf')
    
    # gap graph
    fig = plt.figure(None,figsize=(16,9))
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
        ax3.plot(points_x, points_y, '.', label=gap_type, linestyle='-', linewidth=2.0, markersize=10.0)
        
    ax3.set_xlim(-0.5,len(res['gaps'])+0.5)
    ax3.legend(loc='upper center',ncol=len(gaps_by_type),bbox_to_anchor=(0.5,1.09))
    fig.savefig(file_prefix+'_gaps.pdf')
    
if __name__ == '__main__':
    import sys
    import os
    import optparse
    import hashlib
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    from tvauction.common import json
        
    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    logging.basicConfig(level=log_level)
    
    parser = optparse.OptionParser()
    parser.set_usage('%prog [options] < result.json')
    parser.add_option('--scenopts', dest='scenopts', type='str', help='the options file used to generate the scenarios')
    parser.add_option('--offset', dest='offset', type='int', default=None, help='the scenario offset')
    parser.add_option('--add-prefix', dest='add_prefix', type='str', help='anything else you would like to add to the graph filenames')
    parser.add_option('--graph-path', dest='graph_path', type='str', default='/tmp/tvauction_graphs', help='the base directory for the graphs. has to exist.')
    parser.epilog = 'If scenopts and offset is not passed, the first characters of the md5 of the scenario will be used'
    for option in parser.option_list: 
        if option.default != ("NO", "DEFAULT"): option.help += (" " if option.help else "") + "[default: %default]"
    if sys.stdin.isatty():
        print parser.format_help()
        sys.exit()
    options = parser.parse_args(sys.argv)[0]
    result = sys.stdin.read()
    
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
                int(1 + options.offset / len(generated_options['distributions']))
            ]
            used_distribution = generated_options['distributions'][options.offset-used_random_seed]
            graph_file_scen_prefix = '%s_%s' % ('-'.join(map(str,used_distribution)),used_random_seed)
            graph_file_prefix = '%s_%s' % (graph_file_scen_prefix,graph_file_prefix)
    

    
    
    
    
    drawResult(options.graph_path+'/run_%s' % graph_file_prefix, result)
    