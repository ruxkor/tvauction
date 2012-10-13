
def drawResult(file_prefix, res):

    fig = plt.figure(None,figsize=(20,6))
    
    # the first graph shows all winners and how much they paid
    ind = np.array(sorted(res['winners']))
    width = 0.8
    
    ax1 = fig.add_subplot(111)
    ax1.grid(True,axis='y')
    
    bars = []
    bar_labels = []
    
    for (nr,(ptype,pcolor)) in enumerate(zip(['raw','vcg','core','final'],[(0,1,1),(0,0,0),(1,0,1),(1,1,0)])):
        bar_width = width - nr*width*0.2
        vals = [v for (k,v) in sorted(res['prices_%s' % ptype].iteritems()) if k in ind]
        bar = ax1.bar(ind-bar_width*0.5, vals, bar_width, color=pcolor,linewidth=0.5)
        bars.append(bar)
        bar_labels.append(ptype)

    ax1.set_xticks(ind)
    ax1.set_xticklabels(ind)
    ax1.legend(bars, bar_labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.12))
    
    fig.savefig(file_prefix+'_1.pdf')

    # the second graph is the step info graph    
    steps_info = res['step_info']
    fig = plt.figure(None,figsize=(16,9))
    ax2 = fig.add_subplot(111)
    ax2.grid(True,axis='y')
    
    step_max = 0
    tuples_all = {}
    for what in ('raw','vcg','sep','blocking_coalition','ebpo'):
        tuples_all[what] = tuples_what = [(nr, step_info[what]) for (nr, step_info) in enumerate(steps_info) if what in step_info]
        if tuples_what: step_max = max(step_max, *(s for (s,v) in tuples_what))
        
    for what in ('raw','vcg','sep','blocking_coalition','ebpo'):
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
    fig.savefig(file_prefix+'_2.pdf')
    
    
if __name__ == '__main__':
    import sys
    import os
    import optparse
    from tvauction.common import json
        
    log_level = int(os.environ['LOG_LEVEL']) if 'LOG_LEVEL' in os.environ else logging.WARN
    logging.basicConfig(level=log_level)
    
    parser = optparse.OptionParser()
    parser.set_usage('%prog [options] < result.json')
    parser.add_option('--graph-path',dest='graph_path',type='str',help='the base path for the graphs. the path will be appended with the filename (minus file ending)')
    if sys.stdin.isatty():
        print parser.format_help()
        sys.exit()
    for option in parser.option_list: 
        if option.default != ("NO", "DEFAULT"): option.help += (" " if option.help else "") + "[default: %default]"
    options = parser.parse_args(sys.argv)[0]
    result = json.decode(sys.stdin.read())
    
    graph_path = '/tmp/tvauction/%s-%s-%s' % (price_vector, core_algorithm, random_seed)
    if not os.path.exists(graph_path): os.makedirs(graph_path)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    drawResult(graph_path+'/simulation_%s_%s' % (now,''.join(map(str,distribution))), res)
    