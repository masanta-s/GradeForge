# PaperMind shell environment — dot-source before working:  . .\env.ps1
# Keeps every cache, temp file and model download on this drive (nothing on C:).

$Root = $PSScriptRoot
$Cache = Join-Path $Root ".cache"

$dirs = @{
    TEMP                    = "tmp"
    TMP                     = "tmp"
    PIP_CACHE_DIR           = "pip"
    npm_config_cache        = "npm"
    TORCH_HOME              = "torch"
    TRITON_CACHE_DIR        = "triton"
    TORCHINDUCTOR_CACHE_DIR = "inductor"
    XDG_CACHE_HOME          = "xdg"
}
foreach ($name in $dirs.Keys) {
    $path = Join-Path $Cache $dirs[$name]
    New-Item -ItemType Directory -Force -Path $path | Out-Null
    Set-Item -Path "env:$name" -Value $path
}

# HuggingFace models live under models/ (TrOCR, sentence-transformers)
$env:HF_HOME = Join-Path $Root "models\hf"
New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null

# Privacy flags (see plan: "Environment flags for the air-gapped claim").
# HF_HUB_OFFLINE is switched on by setup_env.py once models are downloaded.
$env:LITELLM_LOCAL_MODEL_COST_MAP = "True"

# Ollama stores models wherever its app is configured (Settings -> model location / OLLAMA_MODELS).

$activate = Join-Path $Root ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) { . $activate }

Write-Host "PaperMind env ready - caches in $Cache, HF models in $env:HF_HOME"
