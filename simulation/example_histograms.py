import numpy as np
import scipy as sp
import random
import scipy.stats as stats
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rc('font', family='serif')
matplotlib.rc('text', usetex='true')

def _findNearest(ar, val):
   idx = (np.abs(ar-val)).argmin()
   return ar[idx]
   
slot_price_steps = [1.0,2.0,5.0,10.0,20.0,50.0,75.0]

def minmax(data,minval,maxval):
    return max(min(data,maxval),minval)
    

res = [_findNearest(slot_price_steps, stats.scoreatpercentile(slot_price_steps,minmax(random.gauss(50,25),0,100))) for _i in xrange(10000)]


reaches = [150000, 150000, 120000, 130000, 310000, 420000, 390000, 390000, 420000, 390000, 430000, 560000, 590000, 650000, 660000, 640000, 730000, 720000, 670000, 660000, 910000, 960000, 890000, 870000, 1010000, 3390000, 2250000, 1390000, 1210000, 840000, 680000, 450000, 400000, 320000, 250000, 230000, 180000, 230000, 250000, 230000, 230000, 260000, 290000, 460000, 570000, 650000, 670000, 730000, 620000, 440000, 840000, 820000, 910000, 720000, 1180000, 1290000, 1530000, 1570000, 1010000, 570000, 480000, 420000, 200000, 160000, 120000, 110000, 120000, 140000, 270000, 140000, 260000, 240000, 260000, 250000, 250000, 330000, 570000, 400000, 580000, 790000, 790000, 740000, 730000, 770000, 1210000, 1340000, 1140000, 1790000, 1880000, 2360000, 1390000, 790000, 490000, 420000, 280000, 210000, 200000, 210000, 200000, 210000, 210000, 180000, 190000, 170000, 220000, 310000, 290000, 340000, 300000, 340000, 270000, 310000, 460000, 550000, 540000, 680000, 870000, 900000, 770000, 750000, 980000, 1280000, 1520000, 1190000, 1950000, 2310000, 1950000, 1170000, 900000, 700000, 590000, 490000, 350000, 290000, 290000, 300000, 270000, 250000, 320000, 310000, 300000, 410000, 330000, 350000, 390000, 400000, 350000, 370000, 480000, 690000, 650000, 830000, 1010000, 910000, 990000, 900000, 870000, 1030000, 1330000, 1510000, 1310000, 1940000, 1430000, 1530000, 1060000, 560000, 420000, 370000, 290000, 300000, 260000, 210000, 170000, 180000, 170000, 250000, 280000, 340000, 220000, 320000, 330000, 310000, 270000, 250000, 340000, 660000, 710000, 640000, 810000, 890000, 820000, 670000, 800000, 1090000, 1110000, 990000, 1630000, 1810000, 1230000, 1170000, 780000, 580000, 480000, 420000, 260000, 190000, 170000, 190000, 160000, 180000, 190000, 210000, 220000, 290000, 210000, 320000, 330000, 290000, 280000, 400000, 390000, 540000, 530000, 720000, 900000, 830000, 700000, 850000, 830000, 1020000, 1190000, 1190000, 920000, 1580000, 1530000, 1360000, 1320000, 970000, 580000, 410000, 400000, 270000, 210000, 240000, 210000, 210000, 200000, 150000, 170000, 190000, 230000, 360000, 320000, 410000, 330000, 370000, 420000, 470000, 550000, 520000, 600000, 500000, 480000, 480000, 600000, 720000, 550000, 430000, 740000, 920000, 1110000, 1100000, 1190000, 2890000, 1730000, 1190000, 1200000, 880000, 720000, 470000, 330000, 310000, 250000, 200000, 180000, 160000, 160000, 220000, 280000, 330000, 410000, 520000, 690000, 660000, 620000, 780000, 1620000, 1920000, 2060000, 1280000, 890000, 990000, 1460000, 1580000, 1850000, 1280000, 660000, 470000, 360000, 500000, 240000, 170000, 110000, 120000, 160000, 230000, 230000, 240000, 350000, 330000, 420000, 440000, 380000, 270000, 390000, 470000, 710000, 520000, 580000, 830000, 800000, 740000, 620000, 900000, 1280000, 1450000, 1130000, 1680000, 1860000, 2180000, 1280000, 630000, 450000, 340000, 220000, 190000, 130000, 100000, 90000, 100000, 90000, 70000, 80000, 130000, 110000, 210000, 160000, 310000, 270000, 350000, 260000, 310000, 340000, 580000, 450000, 590000, 1040000, 1030000, 1010000, 760000, 940000, 1300000, 1350000, 1240000, 1900000, 2220000, 1910000, 1190000, 1010000, 850000, 610000, 510000, 350000, 230000]

fig = plt.figure(figsize=(12,4))

from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ax = fig.add_subplot(1, 2, 1, title="reach - histogram")
ax.hist(reaches, bins=20,color=(0.5,0.5,1.0))
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.set_ylabel('quantity')
ax.set_xlabel('reach')

ax = fig.add_subplot(1, 2, 2, title="$r_i$ - histogram")
ax.hist(res, bins=20, color=(0.5,0.5,1.0))
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax.set_ylabel('quantity')
ax.set_xlabel('$r_i$')

fig.savefig('/tmp/bla.svg', bbox_inches='tight')
