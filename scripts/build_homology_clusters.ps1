param(
    [string]$SourceFastaGz = "",
    [string]$MmseqsBinary = "artifacts/multimodal/tools/mmseqs/bin/mmseqs",
    [string]$WslDistro = "docker-desktop",
    [int]$Threads = 8,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Processed = Join-Path $RepositoryRoot "data/processed"
$Tools = Join-Path $RepositoryRoot "artifacts/multimodal/tools"
$Temp = Join-Path $RepositoryRoot "artifacts/multimodal/tmp/mmseqs"
$InputFasta = Join-Path $Processed "reviewed_human.fasta"
$ClusterTsv = Join-Path $Processed "human40_cluster.tsv"
$ClusterCsv = Join-Path $Processed "homology_clusters.csv"
$MetadataJson = Join-Path $Processed "homology_cluster_audit.json"

function Find-SourceFasta([string]$Start) {
    $Current = Get-Item -LiteralPath $Start
    while ($null -ne $Current) {
        $Candidate = Join-Path $Current.FullName "datasets/02_uniprot/uniprot_reviewed_human.fasta.gz"
        if (Test-Path -LiteralPath $Candidate) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
        $Current = $Current.Parent
    }
    throw "Could not locate uniprot_reviewed_human.fasta.gz; pass -SourceFastaGz."
}

if (-not $SourceFastaGz) {
    $SourceFastaGz = Find-SourceFasta $RepositoryRoot
} else {
    $SourceFastaGz = (Resolve-Path -LiteralPath $SourceFastaGz).Path
}

New-Item -ItemType Directory -Force $Processed, $Tools, $Temp | Out-Null
$Archive = Join-Path $Tools "mmseqs-linux-avx2.tar.gz"
$WindowsBinary = Join-Path $RepositoryRoot ($MmseqsBinary -replace "/", "\")
if (-not (Test-Path -LiteralPath $WindowsBinary)) {
    curl.exe -L --fail --retry 3 -o $Archive https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz
    if ($LASTEXITCODE -ne 0) {
        throw "MMseqs2 download failed."
    }
    tar -xzf $Archive -C $Tools
    if ($LASTEXITCODE -ne 0) {
        throw "MMseqs2 extraction failed."
    }
}

if ($Force -or -not (Test-Path -LiteralPath $InputFasta)) {
    $Input = [System.IO.File]::OpenRead($SourceFastaGz)
    $Output = [System.IO.File]::Create($InputFasta)
    $Gzip = New-Object System.IO.Compression.GZipStream(
        $Input,
        [System.IO.Compression.CompressionMode]::Decompress
    )
    try {
        $Gzip.CopyTo($Output)
    } finally {
        $Gzip.Dispose()
        $Output.Dispose()
        $Input.Dispose()
    }
}

Push-Location $RepositoryRoot
try {
    $LinuxBinary = "./" + ($MmseqsBinary -replace "\\", "/")
    $Version = (& wsl -d $WslDistro -- sh -lc "$LinuxBinary version").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "MMseqs2 could not run in WSL distribution '$WslDistro'."
    }

    if ($Force) {
        Get-ChildItem -LiteralPath $Processed -Filter "human40_*" -ErrorAction SilentlyContinue |
            Remove-Item -Force
        if (Test-Path -LiteralPath $Temp) {
            Remove-Item -LiteralPath $Temp -Recurse -Force
            New-Item -ItemType Directory -Force $Temp | Out-Null
        }
    }

    if (-not (Test-Path -LiteralPath $ClusterTsv)) {
        $Command = @(
            $LinuxBinary,
            "easy-cluster",
            "data/processed/reviewed_human.fasta",
            "data/processed/human40",
            "artifacts/multimodal/tmp/mmseqs",
            "--min-seq-id", "0.4",
            "-c", "0.7",
            "--cov-mode", "0",
            "--threads", $Threads
        ) -join " "
        & wsl -d $WslDistro -- sh -lc $Command
        if ($LASTEXITCODE -ne 0) {
            throw "MMseqs2 clustering failed."
        }
    }
} finally {
    Pop-Location
}

$Clusters = Import-Csv $ClusterTsv -Delimiter "`t" -Header cluster_id,accession
$Clusters | Select-Object accession,cluster_id |
    Export-Csv $ClusterCsv -NoTypeInformation -Encoding utf8
$Metadata = [ordered]@{
    mmseqs_version = $Version
    source_fasta = $SourceFastaGz
    source_sha256 = (Get-FileHash $SourceFastaGz -Algorithm SHA256).Hash
    min_sequence_identity = 0.4
    minimum_coverage = 0.7
    coverage_mode = 0
    sequence_count = $Clusters.Count
    cluster_count = ($Clusters.cluster_id | Sort-Object -Unique).Count
}
$Metadata | ConvertTo-Json | Set-Content $MetadataJson -Encoding utf8
$Metadata | Format-List
