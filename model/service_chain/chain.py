import json
import math
import os
from collections import OrderedDict
from typing import List

import networkx as nx
import numpy as np
from scipy import stats

from .component import Component
from .state import State


class Chain(object):
    """
    Representation of Computation Chain as a Graph
    """
    def __init__(self):
        self.components = OrderedDict() # Set of components
        self.states = OrderedDict() # Set of states
    

    def __str__(self)->str:
        graph = self.compute_graph()
        return str(graph.edges.data())


    def init_components(self, init_conf:dict, 
                        budget:List[int], nconf:int=5)->None:
        for component in init_conf:
            self.components[component] = Component(component, nconf)
            for instance in init_conf[component]:
                self.components[component].add_instance(
                    instance, init_conf[component][instance]
                )
        self.states[0] = State('Initial')
        for i in range(1,len(self.components)):
            self.states[i] = State(f'S{i}')
        self.states[len(self.components)] = State('Final')
        for i,comp in enumerate(self.components.values()):
            comp:Component
            comp.specify_state(self.states[i], self.states[i+1])
        
        if budget is not None:
            self.budget = budget
        else: 
            raise ValueError("Budget not specified")


    def reset(self)->None:
        self.__init__()


    def compute_graph(self)->nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.states.keys())
        for comp in self.components.values():
            comp:Component
            graph.add_edge(comp.prev_state.name, comp.next_state.name,
                           weight=(comp.cpu,comp.mem))
        return graph


    def get_adj_matrix(self)->np.ndarray:
        np_array = np.zeros(
            [
                len(self.components),
                len(self.components)
            ],
            dtype=int
            )
        for i,ic in enumerate(self.components.keys()):
            for j,jc in enumerate(self.components.keys()):
                if (self.components[ic].next_state == 
                    self.components[jc].prev_state):
                    np_array[i][j] = 1
        return np_array


    def get_features(self)->np.ndarray:
        np_array = np.zeros([len(self.components),5])
        for idx,comp in enumerate(self.components.values()):
            comp:Component
            comp.compute_resources()
            np_array[idx][0] = comp.cpu/self.budget[0]
            np_array[idx][1] = comp.mem/self.budget[1]
        return np.nan_to_num(stats.zscore(np_array))


    def get_budget_overrun(self)->float:
        tcpu,tmem = 0,0
        bcpu, bmem = self.budget[0], self.budget[1]
        for comp in self.components.values():
            comp:Component
            comp.compute_resources()
            tcpu += comp.cpu
            tmem += comp.mem
        return math.sqrt(
            (max(0,tcpu-bcpu)/bcpu)**2
            +(max(0,tmem-bmem)/bmem)**2
        )


if __name__ == '__main__':
    chain = Chain()
    
    file_path = os.path.join(os.path.dirname(__file__),
                                '../configs/initial_chain3.json')
    init_conf = json.load(open(file_path))
    flavors_file = os.path.join(os.path.dirname(__file__), 
                                '../configs/flavors5.json')
    flavors = dict(json.load(open(flavors_file))).keys()
    chain.init_components(init_conf,flavors,[110,500])

    print(chain.compute_graph())
