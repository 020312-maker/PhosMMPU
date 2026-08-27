param(
    [string]$Python = "python",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",
    [int[]]$Seeds = @(11, 23, 37, 51, 73),
    [int]$Epochs = 200,
    [int]$BatchSize = 512,
    [int]$Patience = 30
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = Join-Path $Repository "artifacts/multimodal/training_runs"

Push-Location $Repository
try {
    & $Python -m src.multimodal.train `
        --config configs/multimodal/deployment.json `
        --seeds $Seeds `
        --epochs $Epochs `
        --patience $Patience `
        --batch-size $BatchSize `
        --objective nnpu `
        --transformer-layers 2 `
        --modality-dropout 0 `
        --device $Device `
        --network-dir data/processed `
        --network-variant complete `
        --output-root $OutputRoot
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    foreach ($Seed in $Seeds) {
        & $Python -m src.multimodal.export_candidates `
            --run-dir (Join-Path $OutputRoot "nnpu_seed$Seed") `
            --top-k 10
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
} finally {
    Pop-Location
}
