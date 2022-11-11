import json
import os
from collections import Counter, OrderedDict

from state import State

flavors_config = os.path.join(os.path.dirname(__file__), 
                                '../configs/flavors.json')


class Component(object):
    """
    Params 
    * name (str)
    : name of the component
    * TTL (int)
    : time to live of the instance
    
    Returns
    * Component object
    
    Methods
    * add_instances(flavor:str, count:int)->None 
    : add n instances of flavor, default n=1
    * remove_instances(flavor:str, count:int)->bool
    : remove n instances of flavor, default n=1
    * get_instances()->list[str] 
    : get list of instances in the component
    * get_resources()->list[int] 
    : get total resources of the component
    * specify_state(prev_state:State, next_state:State)->None 
    : specify position of the component in the pipeline
    * step()->None 
    : update the component after each timestep
    """
    def __init__(self, name:str, prev_state:State=None, 
                next_state:State=None, TTL:int=3)->None:
        self.name = name
        self.flavors = OrderedDict(json.load(open(flavors_config)))
        for flavor in self.flavors:
            self._impl_TTL(flavor)
        self.config = Counter()
        
        self.TTL = TTL
        self.prev_state = prev_state
        self.next_state = next_state


    def __str__(self):
        return self.name
    
    def _impl_TTL(self, flavor:str)->None:
        self.flavors[flavor].insert(0,0)

    def _update_TTL(self)->None:
        for flavor in self.flavors:
            if self.flavors[flavor][0] > 0:
                self.flavors[flavor][0] -= 1
    
    def specify_state(self, prev_state:State, next_state:State)->None:
        self.prev_state = prev_state
        self.next_state = next_state
    
    def step(self)->None:
        self._update_TTL()

    def get_instances(self):
        return list(self.config.elements())

    def add_instances(self, flavor:str, count:int=1)->None:
        assert isinstance(flavor, str)
        assert flavor in self.flavors
        for i in range(count):
            self.config.update([flavor])
        self.flavors[flavor][0] = self.TTL
    
    def remove_instances(self, flavor:str, count:int=1)->bool:
        assert isinstance(flavor, str)
        assert flavor in self.flavors
        if self.flavors[flavor][0] > 0:
            return False
        for _ in range(count):
            if self.config[flavor] > 0:
                self.config.subtract([flavor])
            elif self.config[flavor] == 0:
                self.config.pop(flavor)
            else:
                return False
        return True

    def get_resources(self)->list[int]:
        self.resources = [0,0]
        for flavor, count in self.config.items():
            self.resources[0] += count * self.flavors[flavor][0]
            self.resources[1] += count * self.flavors[flavor][1]
        return self.resources
