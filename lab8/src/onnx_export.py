import sys
import json
import time
from pathlib import Path

import torch
import numpy as np
import onnx
import onnxruntime as ort

from static_quantization import (
    load_model,
    test_loader,
    RESULTS,
    MODELS
)

torch.set_num_threads(1)

model = load_model("teacher")
model.eval()

print("Teacher checkpoint loaded successfully")

onnx_path = MODELS / "teacher_fp32.onnx"

example = torch.randn(1, 1, 28, 28)

print("Exporting model to ONNX")

torch.onnx.export(
    model,
    example,
    str(onnx_path),
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    dynamo=False
)

onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)

print("ONNX model structure validated")

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1

session = ort.InferenceSession(
    str(onnx_path),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

max_error = 0.0
sum_squared_error = 0.0
output_count = 0
correct = 0
agreement = 0
total = 0

print("Validating against full test dataset")

with torch.inference_mode():
    for x, y in test_loader:
        pytorch_output = model(x).numpy()

        onnx_output = session.run(
            ["output"],
            {"input": x.numpy()}
        )[0]

        error = np.abs(pytorch_output - onnx_output)

        max_error = max(
            max_error,
            float(error.max())
        )

        sum_squared_error += float(
            np.square(error.astype(np.float64)).sum()
        )

        output_count += error.size

        onnx_pred = onnx_output.argmax(axis=1)
        pytorch_pred = pytorch_output.argmax(axis=1)

        correct += int(
            (onnx_pred == y.numpy()).sum()
        )

        agreement += int(
            (onnx_pred == pytorch_pred).sum()
        )

        total += len(y)

mse = sum_squared_error / output_count
accuracy = 100 * correct / total

tolerance = 1e-4
passed = max_error <= tolerance

print("Test images:", total)
print("Maximum absolute error:", max_error)
print("MSE:", mse)
print("Tolerance:", tolerance)
print("Validation passed:", passed)
print("ONNX accuracy:", accuracy)

sample = next(iter(test_loader))[0][:1]

x = sample.numpy()
times_onnx = []
times_pytorch = []

with torch.inference_mode():
    for _ in range(30):
        model(sample)
        session.run(["output"], {"input": x})

    for _ in range(100):
        start = time.perf_counter()
        model(sample)
        times_pytorch.append(
            (time.perf_counter() - start) * 1000
        )

        start = time.perf_counter()
        session.run(["output"], {"input": x})
        times_onnx.append(
            (time.perf_counter() - start) * 1000
        )

result = {
    "model": "teacher_fp32_onnx",
    "opset": 17,
    "provider": "CPUExecutionProvider",
    "onnx_version": onnx.__version__,
    "onnxruntime_version": ort.__version__,
    "cpu_threads": 1,
    "test_images": total,
    "accuracy_pct": accuracy,
    "prediction_agreement_pct": 100 * agreement / total,
    "max_absolute_error": max_error,
    "mse": mse,
    "tolerance": tolerance,
    "validation_passed": bool(passed),
    "onnx_size_mb": onnx_path.stat().st_size / 1e6,
    "onnx_median_ms": float(np.median(times_onnx)),
    "onnx_p95_ms": float(np.percentile(times_onnx, 95)),
    "pytorch_median_ms": float(np.median(times_pytorch)),
    "pytorch_p95_ms": float(np.percentile(times_pytorch, 95))
}

result["speedup"] = (
    result["pytorch_median_ms"] /
    result["onnx_median_ms"]
)

with open(RESULTS / "onnx_validation.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))

if not passed:
    raise RuntimeError(
        "ONNX numerical validation failed"
    )

print("TASK 4 COMPLETED SUCCESSFULLY")
