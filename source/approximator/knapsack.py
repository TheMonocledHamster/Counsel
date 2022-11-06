from parser import Parser
from primal_dual import PrimalDual
import sys
import math
from timeit import default_timer as timer
from collections import Counter


icount = Counter()


def compute_norm(data, ncomp, rlim1, rlim2, rlim3):
    conf_norm = [[] for i in range(ncomp)]
    tmp = 0.0
    for i in range(ncomp):
        tmp_lst = data[i]
        icount[int(len(tmp_lst))] += 1
        for item in tmp_lst:
            tmp_c = float(item[0])/rlim1
            tmp_m = float(item[1])/rlim2
            tmp_l = float(item[2])/rlim3
            tmp = math.sqrt(pow(tmp_c, 2) + pow(tmp_m, 2) + pow(tmp_l, 2))
            tmp_bpb = float(item[3])/tmp if tmp != 0 else 0.0

            conf_norm[i].append([tmp_c, tmp_m, tmp_l, item[3], tmp, tmp_bpb])
    return conf_norm


def is_dominated(item1, item2):
    if(float(item2[3]) > float(item1[3])) and (item2[4] < item1[4]):
        return True
    return False


def remove_dominations(data, ncomp):
    for i in range(ncomp):
        tmp_row = data[i]
        tmp_row.sort(key=lambda x: x[3])
        j = 0
        while (j < len(tmp_row)):
            k = j + 1
            while k < len(tmp_row):
                if(is_dominated(tmp_row[j], tmp_row[k])):
                    tmp_row.remove(tmp_row[j])
                k += 1
            j += 1


def knapsack(data, ncomp, core_lim, mem_lim, lat_lim):
    start = timer()
    data = compute_norm(data, ncomp, core_lim, mem_lim, lat_lim)
    remove_dominations(data, ncomp)
    pd = PrimalDual(data, ncomp, core_lim, mem_lim, lat_lim)
    pd.run()
    end = timer()
    return (pd.is_optimal, pd.get_cost, (end - start))


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python knapsack.py <filename> <ncomp> <core_lim> <mem_lim> <lat_lim>")
        sys.exit(1)
    
    fname = sys.argv[1]
    ncomp = int(sys.argv[2])
    rlim1 = int(sys.argv[3])
    rlim2 = int(sys.argv[4])
    rlim3 = int(sys.argv[5])

    knap_parse = Parser(fname, rlim1, rlim2, rlim3, ncomp)
    data = knap_parse.read_file()
    optimal, cost, exec_time = knapsack(data, ncomp, rlim1, rlim2, rlim3)
    print("time_kpd: " + str(exec_time))
    print("profit_kpd: " + str(cost))
    print("Optimal" if optimal else "Not Optimal")
