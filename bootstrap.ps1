#requires -Version 5.1
<#
.SYNOPSIS
    Set up Engram for development in this repo.

.DESCRIPTION
    Three steps:
      1. `pip install -e .` (with optional [all] extras) — makes
         `import engram` and `engram ...` CLI work everywhere.
      2. `engram install-skill --dev` — symlinks
         `src/engram/_skill_files/` into `~/.claude/skills/engram/`
         so edits in this repo propagate live to every Claude Code session.
      3. Run both test suites (smoke + unit) so a freshly-bootstrapped
         install is verified end-to-end before you start using it.

    For end-user installs (not development), see INSTALL_CN.md / INSTALL.md.

.PARAMETER Python
    Python executable to use (default: `python` from PATH).

.PARAMETER NoExtras
    Skip the `[all]` extras (numpy only). By default the script installs
    sentence-transformers, openai and tiktoken too.

.PARAMETER NoSkill
    Only install the package; skip the `engram install-skill --dev` step.

.PARAMETER NoDev
    Copy the skill files instead of symlinking. Use when you cannot enable
    Windows Developer Mode and don't want to run as Admin.

.PARAMETER SkipPip
    Skip the pip install step (only wire the skill).

.PARAMETER NoTest
    Skip the final smoke + unit test step.

.EXAMPLE
    ./bootstrap.ps1
    Default dev install: editable + all extras + symlinked skill + tests.

.EXAMPLE
    ./bootstrap.ps1 -NoExtras -NoDev
    Minimal: numpy only, copied skill (no Dev Mode needed), tests run.

.EXAMPLE
    ./bootstrap.ps1 -SkipPip -NoSkill -NoTest:$false
    Only run the tests against an already-bootstrapped install.
#>
param(
    [string]$Python = "python",
    [switch]$NoExtras,
    [switch]$NoSkill,
    [switch]$NoDev,
    [switch]$SkipPip,
    [switch]$NoTest
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path

Write-Host "─── Engram dev bootstrap ─────────────────────────────────"
Write-Host "  repo    : $RepoRoot"
Write-Host "  python  : $Python"
Write-Host "  extras  : $(if ($NoExtras) { '(none)' } else { '[all]' })"
Write-Host "  skill   : $(if ($NoSkill) { '(skipped)' } elseif ($NoDev) { 'copy' } else { 'symlink (dev)' })"
Write-Host "  tests   : $(if ($NoTest) { '(skipped)' } else { 'smoke + unit' })"
Write-Host ""

# ── 1. pip install -e . ─────────────────────────────────────────────────────
if ($SkipPip) {
    Write-Host "[1/3] skipped pip install (-SkipPip)"
} else {
    $pipTarget = if ($NoExtras) { "$RepoRoot" } else { "$RepoRoot[all]" }
    Write-Host "[1/3] $Python -m pip install -e `"$pipTarget`""
    & $Python -m pip install -e "$pipTarget"
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed (exit $LASTEXITCODE). Fix the error above and re-run."
    }
    Write-Host "      OK"
}
Write-Host ""

# ── 2. wire the skill ───────────────────────────────────────────────────────
if ($NoSkill) {
    Write-Host "[2/3] skipped skill install (-NoSkill)"
    Write-Host "      Run later: $Python -m engram.cli install-skill --dev"
} else {
    $skillArgs = @("-m", "engram.cli", "install-skill", "--force")
    if (-not $NoDev) { $skillArgs += "--dev" }

    Write-Host "[2/3] $Python $($skillArgs -join ' ')"
    & $Python @skillArgs
    if ($LASTEXITCODE -ne 0) {
        throw "install-skill failed (exit $LASTEXITCODE)."
    }
}
Write-Host ""

# ── 3. verify with tests ────────────────────────────────────────────────────
if ($NoTest) {
    Write-Host "[3/3] skipped tests (-NoTest)"
    Write-Host "      Run later: `$env:ENGRAM_EMBEDDER='hash'; $Python tests/smoke_test.py; $Python tests/unit_test.py"
} else {
    $oldEmbedder = $env:ENGRAM_EMBEDDER
    $env:ENGRAM_EMBEDDER = "hash"
    try {
        Write-Host "[3/3] $Python tests/smoke_test.py"
        & $Python (Join-Path $RepoRoot "tests/smoke_test.py")
        if ($LASTEXITCODE -ne 0) {
            throw "smoke_test.py failed (exit $LASTEXITCODE)."
        }
        Write-Host ""
        Write-Host "      $Python tests/unit_test.py"
        & $Python (Join-Path $RepoRoot "tests/unit_test.py")
        if ($LASTEXITCODE -ne 0) {
            throw "unit_test.py failed (exit $LASTEXITCODE)."
        }
        Write-Host ""
        Write-Host "      $Python tests/agents_test.py"
        & $Python (Join-Path $RepoRoot "tests/agents_test.py")
        if ($LASTEXITCODE -ne 0) {
            throw "agents_test.py failed (exit $LASTEXITCODE)."
        }
    } finally {
        if ($null -eq $oldEmbedder) {
            Remove-Item Env:ENGRAM_EMBEDDER -ErrorAction SilentlyContinue
        } else {
            $env:ENGRAM_EMBEDDER = $oldEmbedder
        }
    }
}
Write-Host ""

# ── done ────────────────────────────────────────────────────────────────────
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Verify CLI is on PATH:"
Write-Host "  $Python -m engram.cli install-skill --check"
Write-Host "  $Python -m engram.cli version"
