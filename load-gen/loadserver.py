import os

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import numpy as np

from .arrival_rates.gen_arrivals import slo_bins, dir

class LoadSetter(BaseModel):
    slo: int
    freq: int
    knob: float

class BaseLoad(BaseModel):
    cpu: list
    mem: list

class LoadServer:
    def __init__(self):
        self.router = APIRouter()
        self.metrics = {
            "slo": 0,
            "arrival_rate": 0.0,
            "load": [0,0],
            "done": False
        }
        self.router.add_api_route('/slo', self.set_load, methods=['PUT'])
        self.router.add_api_route('/base', self.set_base, methods=['PUT'])
        self.router.add_api_route('/load', self.get_load, methods=['GET'])

    def set_load(self, load:LoadSetter):
        self.slo = load.slo
        self.metrics['slo'] = self.slo
        self.freq = load.freq
        self.knob = load.knob
        if not self.slo <= max(slo_bins)*1.5:
            raise Exception(f"SLO {self.slo} is too high")
        elif self.slo > max(slo_bins):
            slo_bin = max(slo_bins)
        else:
            slo_bin = min([bin for bin in slo_bins if bin > self.slo])
        self.arrivals = np.load(os.path.join(dir, f'load_{slo_bin}.npy'))[:,1]
        self.arrivals = np.concatenate((self.arrivals, self.arrivals))
        self.ep_gen = iter(self.arrivals)
        self.weight = np.log(self.slo) * self.knob
        self.new_episode()

    def set_base(self, base:BaseLoad):
        base.cpu = np.array(base.cpu)
        base.mem = np.array(base.mem)
        self.base = base
    
    def new_episode(self):
        self.metrics['done'] = False
        try:
            self.episode_lambda = next(self.ep_gen) * self.freq
        except StopIteration:
            self.ep_gen = iter(self.arrivals)
            self.episode_lambda = next(self.ep_gen) * self.freq
        rand_size = np.random.choice([206, 451, 568, 818, 1000, 1212])
        rand_start = np.random.randint(0, len(self.arrivals) - rand_size)
        self.episode_arrivals = self.arrivals[rand_start:rand_start+rand_size]
        self.step_gen = self.step_generator()

    def step_generator(self):
        ratio = self.episode_lambda / sum(self.episode_arrivals)
        for arrival in self.episode_arrivals:
            yield arrival * ratio

    def calc_util(self, lamda):
        cutil = self.weight * self.base.cpu * lamda
        mutil = self.weight * self.base.mem * lamda
        return [np.ndarray.tolist(cutil), np.ndarray.tolist(mutil)]

    def get_load(self):
        if self.metrics['done']:
            self.new_episode()
        try:
            arrival_rate = next(self.step_gen)
            if arrival_rate < 1:
                arrival_rate = 1
            self.metrics["arrival_rate"] = arrival_rate
            self.metrics["load"] = self.calc_util(arrival_rate)
        except StopIteration:
            self.metrics['done'] = True
        return self.metrics


app = FastAPI()
load_server = LoadServer()
app.include_router(load_server.router)

# for slo in [10, 30, 50, 80, 150, 400, 1000, 3000]:
#     freq = int(1e6 / np.log(np.random.randint(int(slo*0.8), int(slo*1.2))))
#     knob = 0.001
#     load_server.set_load(LoadSetter(slo=slo,freq=freq, knob=knob))
#     load_server.set_base(BaseLoad(cpu=[4,8,6], mem=[8,32,16]))
#     print(f"SLO: {slo}")
#     for i in range(3):
#         print(load_server.get_load())
