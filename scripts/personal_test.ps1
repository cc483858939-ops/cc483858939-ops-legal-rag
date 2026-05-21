param(
    [ValidateSet("reset", "ingest", "retrieve", "ask", "eval", "all")]
    [string]$Action = "all",

    [string]$Query = "agent memory"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Manifest = "/app/configs/personal_test_sources.yml"
$EvalSet = "/app/data/eval/personal_test_eval.yml"
$Collection = "personal_test"

Push-Location $RepoRoot
try {
    $env:QDRANT_COLLECTION = $Collection

    function Invoke-Checked {
        & $args[0] $args[1..($args.Count - 1)]
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
        }
    }

    function Start-Qdrant {
        Invoke-Checked docker compose up -d qdrant
    }

    function Reset-Collection {
        Start-Qdrant
        try {
            Invoke-RestMethod -Method Delete -Uri "http://localhost:6333/collections/$Collection" | Out-Null
            Write-Host "Deleted Qdrant collection: $Collection"
        }
        catch {
            Write-Host "Collection $Collection did not exist or could not be deleted; continuing."
        }
    }

    function Ingest-PersonalTest {
        Start-Qdrant
        Invoke-Checked docker compose run --rm -e QDRANT_COLLECTION=$Collection cli legal-rag ingest --manifest $Manifest
    }

    function Retrieve-PersonalTest {
        Start-Qdrant
        Invoke-Checked docker compose run --rm -e QDRANT_COLLECTION=$Collection cli legal-rag retrieve $Query --debug
    }

    function Ask-PersonalTest {
        Start-Qdrant
        Invoke-Checked docker compose run --rm -e QDRANT_COLLECTION=$Collection cli legal-rag ask $Query
    }

    function Eval-PersonalTest {
        Start-Qdrant
        Invoke-Checked docker compose run --rm -e QDRANT_COLLECTION=$Collection cli legal-rag eval-retrieval --eval-set $EvalSet --manifest $Manifest
    }

    switch ($Action) {
        "reset" { Reset-Collection }
        "ingest" { Ingest-PersonalTest }
        "retrieve" { Retrieve-PersonalTest }
        "ask" { Ask-PersonalTest }
        "eval" { Eval-PersonalTest }
        "all" {
            Reset-Collection
            Ingest-PersonalTest
            Retrieve-PersonalTest
            Ask-PersonalTest
        }
    }
}
finally {
    Pop-Location
}
