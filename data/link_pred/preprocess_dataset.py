## This script is used to normalize timestamps from 0 to 1

import argparse
import sys
import re
import numpy
import os
import math
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True)
args = parser.parse_args()

tmp = (args.i).split(".")
output_filename = tmp[0] + "_preproc.wel"
print(output_filename)
out_file = open(output_filename, "w")

time_stamp_list = []

with open(args.i) as fp:
    print("Reading from file", args.i)
    for ln in fp:
        tmp_list = []
        tmp_list.append(re.findall(r'\d+', ln))
        if len(tmp_list[0]) == 3:       
            # format: src dst timestamp
            time_stamp_list.append(int(tmp_list[0][2]))
        elif len(tmp_list[0]) == 4:     
            # format: src dst smt timestamp (smt = something I don't care about)
            time_stamp_list.append(int(tmp_list[0][3]))
        else:
            # unknown format
            sys.exit("[ERROR] Unknown number of columns in the data set!!")

min_timestamp = min(time_stamp_list)
max_timestamp = max(time_stamp_list)
print("min:", min_timestamp, "max:", max_timestamp)
max_min_diff  = int(max_timestamp) - int(min_timestamp)
print("diff:", max_min_diff)

with open(args.i) as fp:
    print("Writing to file", output_filename)
    for ln in fp:
        tmp_list = []
        tmp_list.append(re.findall(r'\d+', ln))
        src_node = tmp_list[0][0]
        dst_node = tmp_list[0][1]
        if len(tmp_list[0]) == 3:
            # format: src dst timestamp
            time_stamp = round((int(tmp_list[0][2]) - int(min_timestamp)) / int(max_min_diff), 10)
        elif len(tmp_list[0]) == 4:
            # format: src dst smt timestamp (smt = something I don't care about)
            time_stamp = round((int(tmp_list[0][3]) - int(min_timestamp)) / int(max_min_diff), 10)
        else:
            # unknown format
            sys.exit("[ERROR] Unknown number of columns in the data set!!")
        out_file.write(str(src_node) + " " + str(dst_node) + " " + str(time_stamp) + "\n")

out_file.close()