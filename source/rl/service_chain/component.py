import json
import math
import os
import time
from collections import Counter, OrderedDict

from .state import State

flavors_config = os.path.join(os.path.dirname(__file__), 
                                '../../configs/flavors.json')


class Component(object):
    """
    Params 
    * name (str)
    : name of the component
    * TTL (int)
    : time to live in minutes
    
    Returns
    * Component object
    
    Methods
    * add_instances(flavor:str, count:int)->None 
    : add n instances of flavor, default n=1
    * remove_instances(flavor:str, count:int)->bool
    : remove n instances of flavor, default n=1
    * get_instances()->list[str] 
    : get list of instances in the component
    * resource_norm(budget:list[int])->float
    : calculate the resource norm of the component
    * specify_state(prev_state:State, next_state:State)->None 
    : specify position of the component in the pipeline
    * check_TTL(flavor:str)->bool
    : check if the flavor is within the TTL
    """
    def __init__(self, name:str, prev_state:State=None, 
                next_state:State=None, TTL:int=15):
        self.name = name
        self.flavors = OrderedDict(json.load(open(flavors_config)))
        self.config = Counter()
        
        self.TTL = TTL
        self.TTL_tracker = {}
        self.prev_state = prev_state
        self.next_state = next_state


    def __str__(self):
        return self.name

    def check_TTL(self,flavor)->bool:
        if flavor in self.TTL_tracker:
            if time.time() - self.TTL_tracker[flavor] < self.TTL * 60:
                return True
            else:
                self.TTL_tracker.pop(flavor)
                return False
        else:
            return False
    
    def specify_state(self, prev_state:State, next_state:State)->None:
        self.prev_state = prev_state
        self.next_state = next_state

    def get_instances(self):
        return list(self.config.elements())

    def add_instances(self, flavor:str|list[str], count:list=None)->None:
        if isinstance(flavor, str):
            assert flavor in self.flavors
            self.config.update([flavor])
            self.TTL_tracker[flavor] = time.time()
        elif isinstance(flavor, list):
            for f,c in zip(flavor,count):
                assert f in self.flavors
                for _ in range(c):
                    self.config.update([f])
                    self.TTL_tracker[f] = time.time()
    
    def remove_instances(self, flavor:str|list[str], count:list=None)->bool:
        if isinstance(flavor, str):
            assert flavor in self.flavors
            if self.check_TTL(flavor):
                return False
            elif self.config[flavor] > 1:        
                self.config.subtract([flavor])
            elif self.config[flavor] == 1:
                self.config.subtract([flavor])
            else:
                return False
        elif isinstance(flavor, list):
            for f,c in zip(flavor,count):
                assert f in self.flavors
                if self.check_TTL(f):
                    continue
                for _ in range(c):
                    if self.config[f] > 1:
                        self.config.subtract([f])
                    elif self.config[f] == 1:
                        self.config.pop(f)
                else:
                    return False
        return True

    def resource_norm(self, budget:list[int])->float:
        self.cpu, self.mem = 0,0
        for flavor, count in self.config.items():
            self.cpu += count * self.flavors[flavor][0]
            self.mem += count * self.flavors[flavor][1]
        return (math.sqrt( (self.cpu/budget[0])**2
                        + (self.mem/budget[1])**2 ))


if __name__ == '__main__':
    c = Component('test')
    c.add_instances('small', 3)
    c.add_instances('medium', 2)
    c.add_instances('large', 1)
    print(c.get_instances())
    print(c.resource_norm([100,120]))
    print(c.resource_norm([120,100]))
