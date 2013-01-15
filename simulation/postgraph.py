import sys
import re
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import math
from contextlib import closing

from itertools import izip
import csv
import pylab

matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex='true')

cdict = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.3, 0.8, 0.8),
        (0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0)
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (0.3, 0.8, 0.8),
        (0.5, 1.0, 1.0),
        (0.7, 0.8, 0.8),
        (1.0, 0.0, 0.0)
    ),
    'blue': (
        (0.0, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.7, 0.8, 0.8),
        (1.0, 0.0, 0.0),
    ),
}

my_cm = matplotlib.colors.LinearSegmentedColormap('my_cm',cdict,400)

cdict = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.3, 0.1, 0.1),
        (0.5, 1.0, 1.0),
        (0.7, 1.0, 1.0),
        (1.0, 1.0, 1.0)
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (0.3, 0.1, 0.1),
        (0.5, 1.0, 1.0),
        (0.7, 0.4, 0.4),
        (1.0, 0.2, 0.2)

    ),
    'blue': (
        (0.0, 0.2, 0.2),
        (0.3, 0.4, 0.4),
        (0.5, 1.0, 1.0),
        (0.7, 0.4, 0.4),
        (1.0, 0.2, 0.2)
    ),
}

my_cm = matplotlib.colors.LinearSegmentedColormap('my_cm_mono',cdict,5)

matplotlib.colors
def grouped(iterable, n):
    return izip(*[iter(iterable)]*n)

def intelliReplace(header):
    res = header.replace('_',' ')
    if 'vals ' in res:
        res = res.replace('vals ','')
        res = re.sub(r' \d*$','', res)
        res = re.sub(r'^(\w*) (\w*) (\w*)$', r'\1 / \2 (\3)', res)
        res = re.sub(r'^(\w*) (\w*)$', r'\1 / \2', res)
        res = res.replace('bid','$b_j$').replace('vcg','$\pi_j^{vcg}$').replace('final','$\pi_j$')
        res = res.replace('$ / $',' / ')
    else:
        res = re.sub(r'^(\w*) (\w*) (\w*)$', r'\1  \2 (\3)', res)
    print header, res
    return res

def graph(file_paths):
    for file_path in file_paths:
        with open(file_path,'r') as fh:
            file_path = re.sub('^(.*)\..*', r'\1', file_path)

            dr = csv.reader(fh)
            headers = dr.next()
            
            data = []
            for line in dr:
                data.append(map(float,line))
            data = np.array(data)
            
            def boxplot(nr, qty):
                fig = plt.figure(None,figsize=(10,3))
                for (inr,header) in enumerate(headers[nr:nr+qty]):
                    header = intelliReplace(header)
                    datum = data[:,nr+inr]
                    ax = fig.add_subplot(1, qty, inr+1)
                    ax.boxplot(datum)
                    ax.locator_params('y', nbins=6)
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim((ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin)))
                    ax.set_xticks([1])
                    ax.set_xticklabels([header])
                return fig
            
            def heatmap(nr, qty, quantiles):
                fig = plt.figure(figsize=(12,4))
                for (inr, header) in enumerate(headers[nr:nr+(quantiles*qty):quantiles+1]):
                    header = intelliReplace(header)
                    datum = data[:,nr+inr*quantiles:nr+(inr+1)*quantiles]
                    ax = fig.add_subplot(1, qty, inr+1, title=header)
                    im = ax.imshow(datum, aspect=1.5, vmin=-0.5, vmax=0.5, cmap=my_cm, interpolation='nearest')
                    ax.set_yticks([])
                    ax.set_xticks([quantiles/4, quantiles/2, 3*quantiles/4])
                    ax.set_xticklabels(['q25','median','q75'])
                fig.subplots_adjust(left=0.1,bottom=0.15)
                cax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
                fig.colorbar(im,cax=cax,orientation='horizontal')
                return fig
            
            nr = 0
            #   
            # revenue, val_coalition
            fig = boxplot(nr, 2)
            fig.savefig('%s_%s.svg' % (file_path, 'revs'), bbox_inches='tight')
            nr += 2
            #
            # iterations
            nr += 1
            
            # runtime, switches
            fig = boxplot(nr, 2)
            fig.savefig('%s_%s.svg' % (file_path, 'iters'), bbox_inches='tight')
            nr += 2
            #
            # gwd_gap, sep_gap_last
            # vcg_gap_mean, sep_gap_mean
            nr += 4
            
            # vals_bid_final_median, vals_bid_vcg_median, vals_final_vcg_median
            fig = boxplot(nr, 3)
            fig.savefig('%s_%s.svg' % (file_path, 'medians_bid_final_vcg'), bbox_inches='tight')
            nr += 3

            # 50 x vals_final_bid 0..100
            # 50 x vals_final_vcg 0..100
            fig = heatmap(nr, 3, 50)
            fig.savefig('%s_%s.svg' % (file_path, 'vals_bid_ratios'), bbox_inches='tight')
            nr += 2*50
            
            # 50 x vals_final_vcg 0..100
            nr += 50
 
if __name__ == '__main__':
#    file_paths = sys.argv[1:]
    file_paths = ['/tmp/pa_reuse_trim.csv','/tmp/pa_reuse_reuselong.csv']
    graph(file_paths)   
#    plt.show()
