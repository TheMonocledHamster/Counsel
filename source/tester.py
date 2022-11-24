import json
from typing import Tuple
import urllib.request as request

def call_load_server(choice: int)->Tuple:
    """
    Call the server with the action and get the next metrics.
    """
    url = "http://localhost:8000/"
    data = json.dumps({"choice": choice}).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    with request.urlopen(req) as f:
        metrics = json.loads(f.read().decode("utf-8"))
    return (metrics["arrival_rate"], metrics["utilization"],
            metrics["latency"], metrics["action_type"], 
            metrics["action_component"])
