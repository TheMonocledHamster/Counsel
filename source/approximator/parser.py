import csv

class Parser:
    def __init__(self, fname, num, lim_core, lim_mem, lim_lat):
        self.fname = fname
        self.lim_core = float(lim_core)
        self.lim_mem = float(lim_mem)
        self.lim_lat = float(lim_lat)
        self.ncomp = int(num)

    def read_file(self):
        data = [[] for i in range(self.ncomp)]
        cur_comp = 0
        with open(self.fname) as fptr:
            fdata = csv.reader(fptr, delimiter='\t')
            for drow in fdata:
                if any("-" in dr for dr in drow) is False:
                    data[cur_comp].append(drow)
                else:
                    cur_comp += 1
        return data
