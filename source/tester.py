import json
from typing import List
import urllib.request as request
import numpy as np

def set_slo(slo:int):
    """
    Set the SLO for the server.
    """
    url = "http://localhost:8000/"
    data = json.dumps({"slo": slo}).encode("utf-8")
    request.Request(url, data=data, method="PUT")

def call_load_server(choice:int, 
                     cpu:List[int], 
                     mem:List[int]
                     )->List[
                             List[int],
                             np.array[float],
                             float,
                             int,
                             int
                            ]:
    """
    Call the server with the action and get the next metrics.
    """
    url = "http://localhost:8000/"
    data = json.dumps({"choice": choice}).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    with request.urlopen(req) as f:
        metrics = json.loads(f.read().decode("utf-8"))
    
    ucpu = np.array(metrics["resources"][0])
    umem = np.array(metrics["resources"][1])
    util = np.sqrt(np.sum((ucpu/cpu)**2) + np.sum((umem/mem)**2))
    
    return [metrics["arrival_rate"], util, metrics["latency"], 
            metrics["action_type"], metrics["action_component"]]
