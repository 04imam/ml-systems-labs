$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$scripts = @(
    "src/manual_quantization.py",
    "src/dynamic_quantization.py",
    "src/static_quantization.py",
    "src/onnx_export.py",
    "src/combined_optimization.py",
    "project/train_baseline.py"
)

foreach ($script in $scripts) {
    python -u $script
    if ($LASTEXITCODE -ne 0) {
        throw "Execution failed: $script"
    }
}

Write-Host "LAB 8 COMPLETED SUCCESSFULLY"
