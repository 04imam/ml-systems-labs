# Lab 5: Frameworks and Automatic Differentiation

## Objective

The objective of this lab is to implement a scalar automatic differentiation engine, validate gradient calculations against PyTorch, train a small neural network, and compare framework execution performance.

## Tasks 1–3: Custom Automatic Differentiation Engine

A custom `Value` class was implemented to store scalar data, gradients, parent nodes, and local derivative rules.

The engine supports addition, multiplication, powers, ReLU, and reverse-mode automatic differentiation using topological sorting.

The test expression was:

`((a * b + c) ** 2).relu().sum()`

With inputs `a = 2`, `b = 3`, and `c = 1`, the results were:

| Variable | Gradient |
|---|---:|
| a | 42.0 |
| b | 28.0 |
| c | 14.0 |

The output value was 49.0.

## Task 4: Gradient Verification

The custom engine was compared against PyTorch autograd and central finite-difference numerical gradients.

| Method | Gradient a | Gradient b | Gradient c |
|---|---:|---:|---:|
| Custom engine | 42.0 | 28.0 | 14.0 |
| PyTorch | 42.0 | 28.0 | 14.0 |
| Numerical | 42.0000000009 | 28.0000000004 | 13.9999999995 |

Maximum absolute difference: **8.968754627858289e-10**

The result satisfies the required tolerance of `1e-6`.

## Task 5: Neural Network Training

A neural network with architecture 2 → 8 → 1 was trained on the `make_moons` dataset using a sigmoid output and binary cross-entropy loss.

The PyTorch baseline used 200 samples and 300 epochs.

- Mean training accuracy: **99.7%**
- Median training time across five runs: **0.1282 seconds**

A separate controlled benchmark compared the custom scalar autograd engine against PyTorch using identical starting parameters, 60 samples, 60 epochs, and five runs.

| Implementation | Median training time | Final reported loss |
|---|---:|---:|
| Custom engine | 2.1963 s | 0.423387 |
| PyTorch | 0.0120 s | 0.423387 |

PyTorch was approximately **183.66 times faster** in this benchmark.

This performance difference reflects the overhead of constructing and traversing scalar computation graphs in Python compared with optimized tensor operations in PyTorch.

## Task 6: Framework Execution and Tensor Contiguity

A small PyTorch model was inspected using `torch.jit.trace`, including its computation graph and input tensor properties.

Input tensor:

- Shape: `[1024, 2]`
- Dtype: `torch.float32`
- Contiguous: `True`

### Eager vs. Traced Execution

| Execution mode | Median time |
|---|---:|
| Eager | 0.043960 ms/op |
| Traced | 0.034972 ms/op |

Traced execution was **1.26× faster** in this experiment.

### Tensor Contiguity

| Layout | Median operation time |
|---|---:|
| Contiguous | 0.184037 ms/op |
| Non-contiguous | 0.223958 ms/op |

The non-contiguous tensor operation required approximately **1.22× the execution time** of the contiguous operation.

### When Does Compilation or Tracing Help?

Tracing or compilation can improve performance when the same computation is executed repeatedly and Python dispatch overhead is significant. Benefits depend on the model, execution environment, and workload. Compilation also introduces an initial overhead, which should be amortized through repeated execution.

Tensor contiguity influences memory access patterns and operation speed. In this experiment, the non-contiguous tensor was slower for the tested reduction.

## Conclusion

The lab demonstrated computation-graph construction, gradient accumulation, reverse-mode automatic differentiation, and numerical verification.

The custom engine produced gradients consistent with PyTorch and numerical differentiation. The neural-network comparison demonstrated the substantial performance advantage of PyTorch's optimized tensor operations. Traced execution improved inference performance for the tested small model, and tensor contiguity affected reduction speed.

All six implementation and benchmarking tasks were executed successfully.