class PrimalDual:
    def __init__(self, data, ncomp, core_lim, mem_lim, lat_lim):
        self.beta = []
        self.core = float(core_lim)
        self.mem = float(mem_lim)
        self.lat = float(lat_lim)
        self.data = data
        self.ncomp = ncomp
        self.delta = []
        self.cost = 0.0
        self.winner = []
        self.core_use = 0.0
        self.mem_use = 0.0
        self.lat_use = 0.0
        self.winner_index = []
        
        for i in range(ncomp):
            dtmp = data[i]
            btmp = [float(x[3]) for x in dtmp]
            self.beta.append(min(btmp))
        
    def run(self):
        self.delta.append((1.0/float(self.core)))
        self.delta.append((1.0/float(self.mem)))
        self.delta.append((1.0/float(self.lat)))

        for i in range(self.ncomp):
            dtmp = self.data[i]
            dtmp.sort(key=lambda x: x[5], reverse=True)
            winner = dtmp[0]
            converge = False
            win_value = 0.0
            while not converge:
                win_value += (float(winner[0]) * float(self.delta[1]) +
                              float(winner[1]) * float(self.delta[2]) +
                              float(winner[2]) * float(self.delta[2]))
                win_value += self.beta[i]

                if(win_value >= float(winner[4])):
                    converge = True
                else:
                    self.delta[0] += 0.1
                    self.delta[1] += 0.1
                    self.delta[2] += 0.1

                    self.beta[i] += 1

            pos = dtmp.index(winner)
            varname = str('x')+str(i+1)+str(pos+1)
            self.winner_index.append(varname)
            self.winner.append(winner)
            self.cost += float(dtmp[pos][3])
            self.core_use += float(dtmp[pos][0])
            self.mem_use += float(dtmp[pos][1])
            self.lat_use += float(dtmp[pos][2])

            self.core -= int(dtmp[pos][0])
            self.mem -= float(dtmp[pos][1])
            self.lat -= float(dtmp[pos][2])

    def get_cost(self):
        return self.cost

    def is_optimal(self):
        if ((self.core_use > 1) or (self.mem_use > 1) or (self.lat_use > 1)):
            return False
        return True
