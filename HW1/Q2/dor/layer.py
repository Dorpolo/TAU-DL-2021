from abc import ABC, abstractclassmethod


class Layer(ABC):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractclassmethod
    def forward_prop(self, input):
        pass

    @abstractclassmethod
    def backward_prop(self, input):
        pass
