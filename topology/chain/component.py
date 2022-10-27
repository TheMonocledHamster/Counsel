from collections import Counter
import json
import os
from collections import OrderedDict

json_file = os.path.join(os.path.dirname(__file__), 'flavors.json')


class Component:
    """
    Representation of each individual component in the service chain (each node in the graph).
    Params 
        name (str): name of the component
        TTL (int): time to live of the component
    Returns
        Component object
    Methods
        add_instance(flavor:str)->None : add an instance to the component
        remove_instance(flavor:str)->None : remove an instance from the component
        get_instances()->list[str] : get the list of instances in the component
        get_resources()->list[int,int] : get the total resources of the component
        utilization(req:list[int,int])->list[float,float] : get the total utilization of the component
    """
    def __init__(self, name:str, TTL:int=3)->None:
        self.name = name
        self.flavors = OrderedDict(json.load(open(json_file))) # list of ordered dict {name:[cpu,mem]}
        self.impl_TTL(TTL)
        self.instances = Counter() # dict of instances (flavor.name, count)


    def __str__(self):
        return self.name
    
    def impl_TTL(self, time:int)->None:
        for flavor in self.flavors():
            self.instances[flavor].insert(0,time)

    def get_instances(self):
        return self.instances.elements()

    def add_instance(self, flavor:str)->None:
        assert isinstance(flavor, str)
        assert flavor in self.flavors
        self.instances.update([flavor])
    
    def remove_instance(self,flavor:str)->None:
        assert isinstance(flavor, str)
        assert flavor in self.flavors
        if self.flavors[flavor][0] > 0:
            return False
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

    def utilization(self,req:list[int,int])->list[float,float]:
        return [req[0]/self.resources[0], req[1]/self.resources[1]]




if __name__ == '__main__':
    c = Component('c1')
    c.add_instance('nano')
    c.add_instance('nano')
    c.add_instance('large')
    c.add_instance('medium')
    c.remove_instance('nano')
    print([i for i in c.get_instances()])
    c.remove_instance('nano')
    print([i for i in c.get_instances()])
    c.remove_instance('nano')
    print([i for i in c.get_instances()])
    print(c.get_resources())
    print(c.utilization([4,8]))
