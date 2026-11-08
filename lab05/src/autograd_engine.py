
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += power * self.data ** (power - 1) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out


    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else Value(other) ** -1)

    def __rtruediv__(self, other):
        return other * self ** -1

    def __rsub__(self, other):
        return other + (-self)

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        import math
        if self.data >= 0:
            s = 1 / (1 + math.exp(-self.data))
        else:
            e = math.exp(self.data)
            s = e / (1 + e)

        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def sum(self):
        return self

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for node in topo:
            node.grad = 0.0

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    a = Value(2.0)
    b = Value(3.0)
    c = Value(1.0)

    result = ((a * b + c) ** 2).relu().sum()
    result.backward()

    print("Output:", result.data)
    print("Gradient a:", a.grad)
    print("Gradient b:", b.grad)
    print("Gradient c:", c.grad)
