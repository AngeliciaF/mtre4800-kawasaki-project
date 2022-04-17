from copy import deepcopy

class Payload():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.r = 0
        self.type = 4
        self.distance = 100001
        self.bounds = None
        self.selected = 0

    def tags(self):
        tag_set = deepcopy(self.__dict__)
        tag_set.pop('bounds')
        tag_set.pop('selected')
        return tag_set
