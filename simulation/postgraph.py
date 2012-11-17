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
my_cm = matplotlib.colors.LinearSegmentedColormap('my_cm',cdict,256)
def grouped(iterable, n):
    return izip(*[iter(iterable)]*n)

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
                fig = plt.figure(None,figsize=(10,5))
                for (inr,header) in enumerate(headers[nr:nr+qty]):
                    datum = data[:,nr+inr]
                    ax = fig.add_subplot(1, qty, inr+1)
                    ax.boxplot(datum)
                    ax.locator_params('y', nbins=6)
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim((ymin-0.05*(ymax-ymin), ymax+0.05*(ymax-ymin)))
                    ax.set_xticks([1])
                    ax.set_xticklabels([header])
                return fig
            
            def heatmap(nr, qty):
                datum = data[:,nr:nr+50]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                im = ax.imshow(datum, shape=datum.shape, vmin=-0.5, vmax=0.5, cmap=my_cm, interpolation='nearest')
                ax.set_yticks([])
                ax.set_xticks([12, 25, 37])
                ax.set_xticklabels(['q25','median','q75'])
                plt.colorbar(im,ax=ax,orientation='horizontal')
                return fig
            
            
            nr = 0
            #   
            # revenue, val_coalition
            fig = boxplot(nr, 2)
            fig.savefig('%s_%s.svg' % (file_path, 'revs'), bbox_inches='tight')
            nr += 2
            #
            # iterations, switches
            fig = boxplot(nr, 2)
            fig.savefig('%s_%s.svg' % (file_path, 'iters'), bbox_inches='tight')
            nr += 2
            #
            # gwd_gap, sep_gap_last
            # vcg_gap_mean, sep_gap_mean
            nr += 4
            #
            # vals_bid_final_median, vals_bid_vcg_median, vals_final_vcg_median
            fig = boxplot(nr, 2)
            fig.savefig('%s_%s.svg' % (file_path, 'medians_bid_final_vcg'), bbox_inches='tight')
            nr += 3

            # 50 x vals_final_bid 0..100
            fig = heatmap(nr, 50)
            fig.savefig('%s_%s.svg' % (file_path, 'vals_bid_final'), bbox_inches='tight')
            nr += 50
            
            # 50 x vals_final_vcg 0..100
            fig = heatmap(nr, 50)
            nr += 50
            fig.savefig('%s_%s.svg' % (file_path, 'vals_bid_vcg'), bbox_inches='tight')
 
            # 50 x vals_final_vcg 0..100
            fig = heatmap(nr, 50)
            nr += 50
            fig.savefig('%s_%s.svg' % (file_path, 'vals_final_vcg'), bbox_inches='tight')
 
if __name__ == '__main__':
#    file_paths = sys.argv[1:]
    file_paths = ['/tmp/pa_short_long.csv','/tmp/pa_trim_reuse.csv'][1:]
    graph(file_paths)   
    plt.show()
