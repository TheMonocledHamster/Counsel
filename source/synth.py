import requests
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

        ucpu = np.array(metrics["resources"][0])
        umem = np.array(metrics["resources"][1])
        util = np.sqrt(np.mean(((ucpu/cpu)**2)+((umem/mem)**2)))

        act_type = 0
        act_comp = -1
        flag = False
        for i in range(len(cpu)):
            if ucpu[i] > cpu[i] or umem[i] > mem[i]:
                flag = True
                act_type = 1
                act_comp = i
            elif ucpu[i] < cpu[i]/2 and umem[i] < mem[i]/2 and act_comp == -1:
                flag = True
                act_comp = i

        latency = arrival_rate * 1.5 / util

        if flag:
            return arrival_rate, util, latency, act_type, act_comp
        
        time.sleep(2)

def test():
    slo = np.random.randint(30, 5000)
    freq = 1000 / max(np.random.randint(int(0.02*slo), int(0.05*slo)),1e-2)
    set_slo(slo, freq)
    call_load_server([5, 5, 5], [8, 8, 8])

if __name__ == "__main__":
    test()
