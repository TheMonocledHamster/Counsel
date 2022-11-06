from data_parser import dparser
from math import ceil
import sys
import csv


def reset_var(max_c, max_m, max_l, min_c, min_m, min_l):
        min_c = 9999
        min_m = 9999
        min_l = 9999

        max_c = 0
        max_m = 0
        max_l = 0


def generate_config(load, data, ncomp):
    config_set = []
    dim = 4
    delim_file = ["-" for i in range(dim)]

    max_csum = 0
    max_msum = 0
    max_lsum = 0

    min_csum = 0
    min_msum = 0
    min_lsum = 0

    min_c = 9999
    min_m = 9999
    min_l = 9999

    max_c = 0
    max_m = 0
    max_l = 0
    max_value = 0
    min_value = 9999

    for i in range(ncomp):
        cdata = data[i]
        for cd in cdata:
            # print("load: " + str(load) + " cd[3]: " + str(cd[3]))
            try:
                val = ceil(float(cd[3]))
                factor = ceil(load/val)
            except ZeroDivisionError:
                continue
            max_value = max(max_value, float(cd[3]))
            min_value = min(min_value, float(cd[3]))
            # print("factor: " + str(factor))
            if factor > 0:
                cfact = int(cd[0]) * factor
                mfact = int(cd[1]) * factor
                lfact = int(cd[2]) * factor
                prof = (load * 100)/(val * factor)
            else:
                cfact = int(cd[0])
                mfact = int(cd[1])
                lfact = int(cd[2])
                prof = (load * 100)/val

            #print("factor(ceil): " + str(ceil(factor)))
            cd_tmp = []
            #print("min_c = {} cfact = {}".format(min_c, cfact))
            min_c = min(min_c, cfact)
            min_m = min(min_m, mfact)
            min_l = min(min_l, lfact)

            #   print("max_c = {} cfact = {}".format(max_c, cfact))
            max_c = max(max_c, cfact)
            max_m = max(max_m, mfact)
            max_l = max(max_l, lfact)

            cd_tmp.append(cfact)
            cd_tmp.append(mfact)
            cd_tmp.append(lfact)
            cd_tmp.append(prof)
            # print("cd_tmp: " + str(cd_tmp))
            config_set.append(cd_tmp)

        config_set.append(delim_file)

        max_csum += max_c
        max_msum += max_m
        max_lsum += max_l

        min_csum += min_c
        min_msum += min_m
        min_lsum += min_l

        #print("max_c = {}, min_c = {}, max_csum ={}, min_csum = {}".format(max_c, min_c, max_csum, min_csum))
        min_c = 9999
        min_m = 9999
        min_l = 9999

        max_c = 0
        max_m = 0
        max_l = 0

        #   config_set.append(ctmp)
            # print(str(cd))

    #print("Max Sum")
    print("max_core: " + str(max_csum))
    print("max_mem: " + str(max_msum))
    print("max_lat: " + str(max_lsum))
    print("max_value: {}".format(max_value))

    #print("Min Sum")
    print("min_core: " + str(min_csum))
    print("min_mem: " + str(min_msum))
    print("min_lat: " + str(min_lsum))
    print("min_value: {}".format(min_value))

    return config_set


fname = sys.argv[1]
ncomp = int(sys.argv[2])
load = int(sys.argv[3])
ofname = sys.argv[4]
print("Load capacity: " + str(load))

fparse = dparser(fname, ncomp)
data = fparse.read_file()
# print("data: " + str(data))

cset = generate_config(load, data, ncomp)

with open(ofname, "w") as csv_file:
    writer = csv.writer(csv_file, delimiter='\t')
    for line in cset:
        writer.writerow(line)

csv_file.close()


#   print("Config: " + str(cset))
#   for cs in cset:
#       print(str(cs))