import csv

class dparser(object):
    def __init__(self, fname, ncomp):
        self.fname = fname
        self.ncomp = int(ncomp)

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
        fptr.close()
        return data

    def dump_data(self):
        print("Number of components: " + str(self.ncomp))
