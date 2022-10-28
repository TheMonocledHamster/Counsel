import os
import json
from collections import OrderedDict
import networkx as nx

from component import Component
from state import State

class Pipeline(object):
    def __init__(self) -> None:
        self.components = OrderedDict() # Set of components
        self.states = OrderedDict() # Set of states

        self._init_components()
        self._init_states()


    def clear_components(self)->None:
        self.components = {}
    

    def _init_components(self,file_path:str='')->None:
        if file_path == '':
            file_path = os.path.join(os.path.dirname(__file__), \
                '../configs/pipeline.json')
        pipeline_config = json.load(open(file_path))
        
        for component in pipeline_config:
            self.components[component] = Component(component)
            for instance in pipeline_config[component]:
                self.components[component].add_instances(instance, \
                    pipeline_config[component][instance])

    def _init_states(self)->None:
        self.states[0] = State('Initial')
        for i in range(1,len(self.components)):
            self.states[i] = State(f'S{i}')
        self.states[len(self.components)] = State('Final')
        for i,comp in enumerate(self.components.values()):
            comp.specify_state(self.states[i], self.states[i+1])


    def generate_graph(self)->nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.states.keys())

        for comp in self.components.values():
            graph.add_edge(comp.prev_state.name, comp.next_state.name, capacity=comp.get_resources())

        return graph




if __name__ == '__main__':
    pipeline = Pipeline()
    for comp in pipeline.components:
        print(comp)
        print(pipeline.components[comp].get_instances())
        print(pipeline.components[comp].get_resources())
        print(pipeline.components[comp].util([4,12]))
        print(pipeline.components[comp].prev_state,pipeline.components[comp].next_state)
        print()
