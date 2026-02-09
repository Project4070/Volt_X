# Volt X — Development helper script
# Usage: .\scripts\dev.ps1 [command]
#
# Commands:
#   check    — cargo check --workspace
#   test     — cargo test --workspace
#   clippy   — cargo clippy --workspace -- -D warnings
#   bench    — cargo bench --workspace
#   doc      — cargo doc --workspace --no-deps --open
#   fmt      — cargo fmt --all
#   all      — run check, clippy, test, fmt in sequence

param(
    [Parameter(Position=0)]
    [string]$Command = "all"
)

function Write-Step($msg) {
    Write-Host "`n=== $msg ===" -ForegroundColor Cyan
}

switch ($Command) {
    "check" {
        Write-Step "cargo check"
        cargo check --workspace
    }
    "test" {
        Write-Step "cargo test"
        cargo test --workspace
    }
    "clippy" {
        Write-Step "cargo clippy"
        cargo clippy --workspace -- -D warnings
    }
    "bench" {
        Write-Step "cargo bench"
        cargo bench --workspace
    }
    "doc" {
        Write-Step "cargo doc"
        cargo doc --workspace --no-deps --open
    }
    "fmt" {
        Write-Step "cargo fmt"
        cargo fmt --all
    }
    "all" {
        Write-Step "cargo fmt --check"
        cargo fmt --all -- --check
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        Write-Step "cargo clippy"
        cargo clippy --workspace -- -D warnings
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        Write-Step "cargo test"
        cargo test --workspace
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        Write-Host "`nAll checks passed!" -ForegroundColor Green
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Available: check, test, clippy, bench, doc, fmt, all"
    }
}
