from collections import Counter
import json
import os
from collections import OrderedDict

json_file = os.path.join(os.path.dirname(__file__), 'flavors.json')


class Component:
    """
    Representation of each individual component in the service chain (each node in the graph).
    """
    def __init__(self, name:str):
        self.name = name
        self.flavors = OrderedDict(json.load(open(json_file))) # list of ordered dict {name:[cpu,mem]}
        self.instances = Counter() # dict of instances (flavor.name, count)
        # self.utilization = [0,0] # [cpu, mem]

    def __str__(self):
        return self.name

    def get_instances(self):
        return self.instances.elements()

    def add_instance(self, flavor:str)->None:
        assert isinstance(flavor, str)
        assert flavor in self.flavors
        self.instances.update([flavor])
    
    def remove_instance(self,flavor:str)->None:
        assert isinstance(flavor, str)
        assert flavor in self.flavors
        if self.instances[flavor] > 0:
            self.instances.subtract([flavor])
        elif self.instances[flavor] == 0:
            self.instances.pop(flavor)
        else:
            return False
        return True

    def get_resources(self)->list[int,int]:
        self.resources = [0,0]
        for flavor, count in self.instances.items():
            self.resources[0] += count * self.flavors[flavor][0]
            self.resources[1] += count * self.flavors[flavor][1]
        return self.resources

    def utilization(self,requirements:list[int,int])->list[float,float]:
        return [requirements[0]/self.resources[0], requirements[1]/self.resources[1]]




if __name__ == '__main__':
    c = Component('c1')
    c.add_instance('tiny')
    c.add_instance('tiny')
    c.add_instance('large')
    c.add_instance('medium')
    c.remove_instance('tiny')
    print([i for i in c.get_instances()])
    c.remove_instance('tiny')
    print([i for i in c.get_instances()])
    c.remove_instance('tiny')
    print([i for i in c.get_instances()])
    print(c.get_resources())
    print(c.utilization([4,8]))
