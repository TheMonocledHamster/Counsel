import requests
import numpy as np
from typing import List, Tuple
from .service_chain.component import Component


def set_slo(slo:int, freq:int, knob:float)->None:
    """
    Set the SLO for the load.
    """
    url = "http://localhost:8000/slo"
    query = {"slo": slo, "freq": freq, "knob": knob}
    requests.put(url, json=query)

def set_base(comps:List[Component])->None:
    """
    Set the base resources for the load.
    """
    cpu = [comp.cpu for comp in comps]
    mem = [comp.mem for comp in comps]
    url = "http://localhost:8000/base"
    query = {"cpu": cpu, "mem": mem}
    requests.put(url, json=query)


def call_load_server(cpu:List[int], mem:List[int])->Tuple:
    """
    Call the server with the action and get the next metrics.
    """
    while True:
        url = "http://localhost:8000/load"
        metrics = requests.get(url).json()

        arrival_rate = metrics["arrival_rate"]
        lcpu = np.array(metrics["load"][0])
        lmem = np.array(metrics["load"][1])
        slo = metrics["slo"]
        done = metrics["done"]

        act_type = 0
        act_comp = -1
        flag = False
        for i in range(len(cpu)):
            if lcpu[i] * 0.85 > cpu[i] or lmem[i] * 0.85 > mem[i]:
                flag = True
                act_type = 1
                act_comp = i
            elif lcpu[i] < cpu[i]*0.5 and lmem[i] < mem[i]*0.5 and not flag:
                flag = True
                act_comp = i

        loadc = [lcpu[i]/(cpu[i]+1e-7) for i in range(len(cpu))]
        loadm = [lmem[i]/(mem[i]+1e-7) for i in range(len(mem))]

        loadc = [min(1, loadc[i]) for i in range(len(loadc))]
        loadc = [max(0, loadc[i]) for i in range(len(loadc))]
        loadm = [min(1, loadm[i]) for i in range(len(loadm))]
        loadm = [max(0, loadm[i]) for i in range(len(loadm))]

        rho = np.sqrt((np.mean(loadc)**2 + np.mean(loadm)**2)/2)

        if rho < 1:
            latency = 0.95 * slo
        else:
            latency = 0.95 * slo * rho

        return arrival_rate, loadc, loadm, latency, act_type, act_comp, done
