#!/usr/bin/env python
# -*- coding: utf-8; -*-

import sys, os
import numpy as np
import csv
from scipy.stats import scoreatpercentile
from contextlib import closing
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from tvauction.common import json, convertToNamedTuples


files=sys.argv[1:]

rows = []
header = [
    'filename','revenue','val_coalition','iterations','switches',
    'gwd_gap','sep_gap_last','vcg_gap_mean','sep_gap_mean',
    'vals_final_bid_median','vals_vcg_bid_median','vals_vcg_final_median'
] 
header += ['vals_final_bid_%d' % pct for pct in range(0,101,2)]
header += ['vals_vcg_bid_%d' % pct for pct in range(0,101,2)]
header += ['vals_vcg_final_%d' % pct for pct in range(0,101,2)]

rows.append(header)
for result_file_path in files:
    with closing(open(result_file_path,'r')) as result:
        res = result.read()
        data = json.decode(res)
        price_diff_final_to_bid = []
        price_diff_vcg_to_bid = []
        price_diff_vcg_to_final = []
        for w in data['winners']:
            price_bid = data['prices_bid'][w]
            price_vcg = data['prices_vcg'][w]
            price_final = data['prices_final'][w]
            price_diff_final_to_bid.append(round(float(price_final) / price_bid,4))
            price_diff_vcg_to_bid.append(round(float(price_vcg) / price_bid,4))
            price_diff_vcg_to_final.append(round(float(price_vcg) / price_final,4))

        vals_final_bid = OrderedDict()
        vals_vcg_bid = OrderedDict()
        vals_vcg_final = OrderedDict()
        for percstore,percread in zip(
            [vals_final_bid, vals_vcg_bid, vals_vcg_final],
            [price_diff_final_to_bid, price_diff_vcg_to_bid, price_diff_vcg_to_final]
        ):
            for perc in range(0,101,2):
                percstore[perc] = round(scoreatpercentile(percread,perc),4)
        row = []
        row.append(os.path.basename(result_file_path))
        row.append(int(sum(data['prices_final'].itervalues())))
        row.append(int(sum(data['prices_bid'].itervalues())))
        row.append(len(data['gaps']))
        row.append(sum(1 for (c,(cdtype,is_best)) in data['coalitions'] if is_best))
        row.append(data['gaps'][0][1])
        row.append(max([(int(gname.split('_')[1]),gap) for (gname,gap) in data['gaps'] if gname.startswith('sep')])[1])
        row.append(np.mean([gap for (gname,gap) in data['gaps'] if gname.startswith('vcg')]))
        row.append(np.mean([gap for (gname,gap) in data['gaps'] if gname.startswith('sep')]))
        row.append(vals_final_bid[50])
        row.append(vals_vcg_bid[50])
        row.append(vals_vcg_final[50])
        row.extend(vals_final_bid.itervalues())
        row.extend(vals_vcg_bid.itervalues())
        row.extend(vals_vcg_final.itervalues())
        rows.append(row)


reswriter = csv.writer(sys.stdout)
reswriter.writerows(rows)
