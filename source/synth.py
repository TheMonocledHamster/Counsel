import requests
import math
import numpy as np
import time
from typing import List, Tuple


def set_slo(slo:int, freq:int):
    """
    Set the SLO for the server.
    """
    url = "http://localhost:8000/slo"
    query = {"slo": slo, "freq": freq}
    requests.put(url, json=query)

def call_load_server(cpu:List[int], mem:List[int])->Tuple:
    """
    Call the server with the action and get the next metrics.
    """
    while True:
        url = "http://localhost:8000/load"
        metrics = requests.get(url).json()

        arrival_rate = metrics["arrival_rate"]

        ucpu = np.array(metrics["util"][0])
        umem = np.array(metrics["util"][1])

        act_type = 0
        act_comp = -1
        flag = False
        for i in range(len(cpu)):
            if ucpu[i] > cpu[i] or umem[i] > mem[i]:
                flag = True
                act_type = 1
                act_comp = i
            elif ucpu[i] < cpu[i]/2 and umem[i] < mem[i]/2 and not flag:
                flag = True
                act_comp = i

        if flag:
            latency = None #TODO
            util = zip([ucpu[i]/cpu[i] for i in range(len(cpu))],
                       [umem[i]/mem[i] for i in range(len(mem))])
            return arrival_rate, util, latency, act_type, act_comp
        
        time.sleep(2)

if __name__ == "__main__":
    slo = int(np.exp(np.random.randint(300,840)/100))
    freq = int(1e6 / np.random.randint(int(slo*0.8), int(slo*1.2)))
    print("SLO: {}, Freq: {}".format(slo, freq))
    set_slo(slo, freq)
    print(call_load_server([5, 8, 5], [6, 8, 12]))
