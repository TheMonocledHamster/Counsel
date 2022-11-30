import requests
import numpy as np
from typing import List, Tuple
from service_chain.component import Component


def set_slo(slo:int, freq:int)->None:
    """
    Set the SLO for the load.
    """
    url = "http://localhost:8000/slo"
    query = {"slo": slo, "freq": freq}
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

        ep_done = metrics["done"]

        act_type = 0
        act_comp = -1
        flag = False
        for i in range(len(cpu)):
            if lcpu[i] > cpu[i] or lmem[i] > mem[i]:
                flag = True
                act_type = 1
                act_comp = i
            elif lcpu[i] < cpu[i]/2 and lmem[i] < mem[i]/2 and not flag:
                flag = True
                act_comp = i

        latency = 10 #TODO
        # load = zip([lcpu[i]/cpu[i] for i in range(len(cpu))],
        #             [lmem[i]/mem[i] for i in range(len(mem))])
        loadc = [max(lcpu[i]/cpu[i],1) for i in range(len(cpu))]
        loadm = [max(lmem[i]/mem[i],1) for i in range(len(mem))]
        return arrival_rate, loadc, loadm, latency, act_type, act_comp


if __name__ == "__main__":
    slo = int(np.exp(np.random.randint(300,840)/100))
    freq = int(1e8 / np.random.randint(int(slo*0.8), int(slo*1.2)))
    print("SLO: {}, Freq: {}".format(slo, freq))
    from env import CustomEnv
    env = CustomEnv("_", 2048, [30,60], slo, 0.2, 'synthetic')
    set_slo(slo, freq)
    set_base(env.components)
    for i in range(15):
        print(env.step(1)[2])
