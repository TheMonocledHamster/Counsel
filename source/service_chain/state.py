class State(object):
    def __init__(self, name:str) -> None:
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
