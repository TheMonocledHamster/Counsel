import json
import math
import os
from collections import OrderedDict

import networkx as nx
import numpy as np
from scipy import stats

from component import Component
from state import State


class Chain(object):
    """
        Representation of Computational Chain as a graph
    """
    def __init__(self, budget:list[int]=None):
        self.components = OrderedDict() # Set of components
        self.states = OrderedDict() # Set of states

    def _clear_components(self)->None:
        self.components = {}
        self.states = {}


    def init_components(self, init_conf:dict, budget:list[int]=None)->None:
        for component in init_conf:
            self.components[component] = Component(component)
            for instance in init_conf[component]:
                self.components[component].add_instances(
                    instance, init_conf[component][instance]
                )
        self.states[0] = State('Initial')
        for i in range(1,len(self.components)):
            self.states[i] = State(f'S{i}')
        self.states[len(self.components)] = State('Final')
        for i,comp in enumerate(self.components.values()):
            comp.specify_state(self.states[i], self.states[i+1])
        
        if budget is not None:
            self.budget = budget
        else: 
            raise ValueError("Budget not specified")

        
        self.graph_repr = self.generate_graph()
        self.adj_matrix = self.get_adj_matrix()
        self.feature_matrix = self.get_features()
    

    def reset(self, budget:list[int])->None:
        self._clear_components()
        self.__init__(budget)


    def generate_graph(self)->nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.states.keys())
        for comp in self.components.values():
            graph.add_edge(comp.prev_state.name, comp.next_state.name,
                           capacity=comp.get_resources())
        return graph
    
    def get_adj_matrix(self)->np.ndarray:
        np_array = np.zeros([len(self.states),len(self.states)],dtype=int)
        for i,ic in enumerate(self.components.keys()):
            for j,jc in enumerate(self.components.keys()):
                if (self.components[ic].next_state == 
                    self.components[jc].prev_state):
                    np_array[i][j] = 1
        return np_array

    def get_features(self)->np.ndarray:
        np_array = np.zeros([len(self.components),len(self.components)])
        for idx,comp in enumerate(self.components.values()):
            np_array[idx][0] = comp.resource_norm(self.budget)
        return np.nan_to_num(stats.zscore(np_array))
    
    def get_budget_overrun(self)->float:
        total_resources = [0,0]
        for comp in self.components.values():
            total_resources[0] += comp.get_resources()[0]
            total_resources[1] += comp.get_resources()[1]
        print(total_resources)
        return (math.sqrt( 
                    (total_resources[0]/self.budget[0])**2
                +   (total_resources[1]/self.budget[1])**2 )
                /math.sqrt(2) - 1)


if __name__ == '__main__':
    chain = Chain(None, [13,54])
    
    file_path = os.path.join(os.path.dirname(__file__),
                                    '../configs/initial_chain.json')
    init_conf = json.load(open(file_path))
    chain.init_components(init_conf)
    
    for comp in chain.components.values():
        print(comp)
        print(comp.get_instances())
        print(comp.get_resources())
        print(comp.prev_state, comp.next_state)
        print()
    print(chain.graph_repr.edges.data())
    print()
    print(chain.get_adj_matrix())
    print()
    print(chain.get_features())
    print()
    print(chain.get_budget_overrun())
