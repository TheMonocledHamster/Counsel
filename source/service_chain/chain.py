import os
import json
from collections import OrderedDict
import networkx as nx
import numpy as np

from component import Component
from state import State

class Chain(object):
    """
        Representation of Computational Chain as a graph
    """
    def __init__(self) -> None:
        self.components = OrderedDict() # Set of components
        self.states = OrderedDict() # Set of states

        self._init_components()
        self._init_states()

        self.graph_repr = self.generate_graph()
        self.adj_matrix = self.get_adj_matrix()
        self.feature_matrix = self.get_features()


    def _clear_components(self)->None:
        self.components = {}
        self.states = {}


    def _init_components(self,file_path:str='')->None:
        if file_path == '':
            file_path = os.path.join(os.path.dirname(__file__),
                                     '../configs/initial_chain.json')
        chain_config = json.load(open(file_path))

        for component in chain_config:
            self.components[component] = Component(component)
            for instance in chain_config[component]:
                self.components[component].add_instances(
                    instance, chain_config[component][instance]
                )

    def _init_states(self)->None:
        self.states[0] = State('Initial')
        for i in range(1,len(self.components)):
            self.states[i] = State(f'S{i}')
        self.states[len(self.components)] = State('Final')
        for i,comp in enumerate(self.components.values()):
            comp.specify_state(self.states[i], self.states[i+1])
    

    def reset(self)->None:
        self._clear_components()
        self.__init__()


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
        # Implement Capacity Logic Here
        raise NotImplementedError
        # Z-Score
        np_array = ((np_array - np.mean(np_array))/np.std(np_array) 
                    + np.finfo(np.float32).eps)
        return np_array


if __name__ == '__main__':
    chain = Chain()
    for comp in chain.components:
        print(comp)
        print(chain.components[comp].get_instances())
        print(chain.components[comp].get_resources())
        print(chain.components[comp].util([4,12]))
        print(chain.components[comp].prev_state, \
            chain.components[comp].next_state)
        print()
        print(chain.get_adj_matrix())
        print()
