$ErrorActionPreference = "Stop"

# 用法:
# powershell -ExecutionPolicy Bypass -File scripts/download_edx_1425617.ps1

$outDir = "data\downloads\edx_1425617"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$urls = @(
  "https://edx.netl.doe.gov/storage/f/edx/2018/03/2018-03-13T15:25:22.261Z/25660524-5937-4657-bbc8-a86eb6e6d163/anode-greyscale.zip",
  "https://edx.netl.doe.gov/storage/f/edx/2018/03/2018-03-13T15:25:22.283Z/0253f0df-5ad0-49b9-983f-ff2773c19884/anode-segmented.zip",
  "https://edx.netl.doe.gov/storage/f/edx/2018/03/2018-03-13T15:25:22.297Z/00fb8920-8159-4fa7-95d2-626b26764209/cathode-greyscale.zip",
  "https://edx.netl.doe.gov/storage/f/edx/2018/03/2018-03-13T15:25:22.306Z/8eed7209-e68b-4232-94cc-0b66c61f43f0/cathode-segmented.zip",
  "https://edx.netl.doe.gov/storage/f/edx/2018/03/2018-03-13T15:25:22.313Z/ab58d286-90e0-4300-b9fa-5be290057f5e/synth-015.zip",
  "https://edx.netl.doe.gov/storage/f/edx/2018/03/2018-03-13T15:25:22.319Z/c516b041-cb46-4832-ba41-ce005fa2b597/synth-035.zip",
  "https://edx.netl.doe.gov/storage/f/edx/2018/03/2018-03-13T15:25:22.325Z/1c0147dc-f9e2-4d1a-a5af-95130047af70/synth-060.zip",
  "https://edx.netl.doe.gov/storage/f/edx/2018/03/2018-03-13T15:25:22.331Z/27a41c86-a36d-4afd-9fea-cb4e9b14fc61/synth-080.zip"
)

$headers = @{
  "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

foreach ($u in $urls) {
  $name = Split-Path $u -Leaf
  $dst = Join-Path $outDir $name
  Write-Host "Downloading: $name"
  try {
    Invoke-WebRequest -Uri $u -Headers $headers -OutFile $dst -TimeoutSec 120
    $size = (Get-Item $dst).Length
    if ($size -lt 1024) {
      Write-Warning "文件过小，疑似被 403/重定向拦截: $name ($size bytes)"
    } else {
      Write-Host "OK: $name ($size bytes)"
    }
  } catch {
    Write-Warning "FAILED: $name => $($_.Exception.Message)"
  }
}

Write-Host "Done. 输出目录: $outDir"
