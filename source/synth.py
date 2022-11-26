import json
from typing import List
import urllib.request as request
import numpy as np
import time

def set_slo(slo:int):
    """
    Set the SLO for the server.
    """
    url = "http://localhost:8000/slo"
    data = json.dumps({"slo": slo}).encode("utf-8")
    request.Request(url, data=data, method="PUT")

def call_load_server(choice:int, 
                     cpu:List[int], 
                     mem:List[int]
                     )->List[
                             List[int],
                             np.array[float],
                             float
                            ]:
    """
    Call the server with the action and get the next metrics.
    """
    while True:
        url = "http://localhost:8000/load"
        data = json.dumps({"choice": choice}).encode("utf-8")
        req = request.Request(url, data=data, method="POST")
        with request.urlopen(req) as f:
            metrics = json.loads(f.read().decode("utf-8"))
        arrival_rate = np.array(metrics["arrival_rate"])
        latency = metrics["latency"]

        ucpu = np.array(metrics["resources"][0])
        umem = np.array(metrics["resources"][1])
        util = np.sqrt(np.sum((ucpu/cpu)**2) + np.sum((umem/mem)**2))

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

        if flag:
            return [arrival_rate, util, latency, act_type, act_comp]
        
        time.sleep(5)
