import math

class Value():
    """ Implement Forward and Backward Passes for operations needed for the classification of Flattened MNIST data"""
    def __init__(self, data, children = [], op = "", label = ""):
        self.data = data
        self.children = set(children)
        self.op = op
        self.grad = 0.0
        self._backward = lambda: None
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, [self, other], "+")

        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad

        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, [self, other], "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self*other
    
    def __pow__(self, other):
        out = Value(self.data**other, [self], "**")

        def _backward():
            self.grad += other*(self.data**(other-1)) * out.grad

        out._backward = _backward
        return out
    
    def __rpow__(self, other):
        return self**other
    
    def __neg__(self):
        return -1*self
    
    def __sub__(self, other):
        return self + -other
    
    def __truediv__(self, other):
        return self*(other**-1)
    
    def __rtruediv__(self, other):
        return self**-1 * other
    
    def exp(self):
        out = Value(math.exp(self.data), [self], "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(self.data), [self], "log")

        def _backward():
            self.grad += self.data**-1 * out.grad

        out._backward = _backward

        return out

    def relu(self):
        x = self.data if self.data >= 0 else 0
        out = Value(x, [self], "relu")

        def _backward():
            if x <= 0:
                self.grad += 0
            else:
                self.grad += out.grad

        out._backward = _backward

        return out
    

    def __repr__(self) -> str:
        return f"{self.data}"

    def backward(self):
        self.grad = 1.0
        members_seen = set()
        topo_list = []

        def create_topo(member):
            if member not in members_seen:
                members_seen.add(member)
                for child in member.children:
                    create_topo(child)
                topo_list.append(member)     
        
        create_topo(self)

        for member in topo_list[::-1]:
            member._backward()

