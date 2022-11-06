import csv

class dparser(object):
    def __init__(self, fname, ncomp):
        self.fname = fname
        self.ncomp = int(ncomp)

    def read_file(self):
        # data[cur_comp] = []
        data = [[] for i in range(self.ncomp)]
        cur_comp = 0
        with open(self.fname) as fptr:
            fdata = csv.reader(fptr, delimiter='\t')
            # data[cur_comp] = []
            # data[cur_comp].append([])
            for drow in fdata:
                if any("-" in dr for dr in drow) is False:
                    # tmp.append(drow)
                    #   print("drow: " + str(drow))
                    data[cur_comp].append(drow)
                    # print(str(drow))
                else:
                    # print("drow limiter: " + str(drow))
                    cur_comp += 1
                    # data[cur_comp] = []

        fptr.close()

        # print("Filename: " + self.fname)
        # print("cur_comp " + str(cur_comp))

        return data

    def dump_data(self):
        print("Number of components: " + str(self.ncomp))
