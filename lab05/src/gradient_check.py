
import torch
from autograd_engine import Value

def engine_gradients(a, b, c):
    x, y, z = Value(a), Value(b), Value(c)
    result = ((x * y + z) ** 2).relu().sum()
    result.backward()
    return [x.grad, y.grad, z.grad]

def torch_gradients(a, b, c):
    x = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(c, dtype=torch.float64, requires_grad=True)

    result = ((x * y + z) ** 2).relu().sum()
    result.backward()
    return [x.grad.item(), y.grad.item(), z.grad.item()]

def numerical_gradients(a, b, c, eps=1e-5):
    def f(values):
        x, y, z = values
        return max(0.0, (x * y + z) ** 2)

    values = [a, b, c]
    gradients = []

    for i in range(3):
        plus = values.copy()
        minus = values.copy()
        plus[i] += eps
        minus[i] -= eps
        gradients.append((f(plus) - f(minus)) / (2 * eps))

    return gradients

if __name__ == "__main__":
    inputs = (2.0, 3.0, 1.0)

    custom = engine_gradients(*inputs)
    pytorch = torch_gradients(*inputs)
    numerical = numerical_gradients(*inputs)

    print("Custom gradients:", custom)
    print("PyTorch gradients:", pytorch)
    print("Numerical gradients:", numerical)

    diff = max(
        max(abs(a - b) for a, b in zip(custom, pytorch)),
        max(abs(a - b) for a, b in zip(custom, numerical))
    )

    print("Maximum absolute difference:", diff)
    assert diff < 1e-6, "Gradient verification failed!"
    print("TASK 4 PASSED")
