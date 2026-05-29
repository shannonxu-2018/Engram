#!/usr/bin/env bash
# bootstrap.sh — set up Engram for development in this repo (Linux/macOS).
#
# Three steps:
#   1. `pip install -e .` (with [all] extras by default) — makes
#      `import engram` and `engram ...` CLI work everywhere.
#   2. `engram install-skill --dev` — symlinks
#      `src/engram/_skill_files/` into `~/.claude/skills/engram/`
#      so edits in this repo propagate live to every Claude Code session.
#   3. Run both test suites (smoke + unit) so a freshly-bootstrapped
#      install is verified end-to-end before you start using it.
#
# For end-user installs (not development), see INSTALL.md / INSTALL_CN.md.
#
# Usage:
#   ./bootstrap.sh                    # default: editable + [all] + symlink + tests
#   PYTHON=python3.11 ./bootstrap.sh  # pick a specific interpreter
#   ./bootstrap.sh --no-extras        # numpy only
#   ./bootstrap.sh --no-dev           # copy skill instead of symlink
#   ./bootstrap.sh --no-skill         # only install the package
#   ./bootstrap.sh --skip-pip         # only wire the skill (and run tests)
#   ./bootstrap.sh --no-test          # skip the final smoke + unit tests

set -euo pipefail

PYTHON="${PYTHON:-python3}"
NO_EXTRAS=0
NO_SKILL=0
NO_DEV=0
SKIP_PIP=0
NO_TEST=0

for arg in "$@"; do
    case "$arg" in
        --no-extras) NO_EXTRAS=1 ;;
        --no-skill)  NO_SKILL=1 ;;
        --no-dev)    NO_DEV=1 ;;
        --skip-pip)  SKIP_PIP=1 ;;
        --no-test)   NO_TEST=1 ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "bootstrap.sh: unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "─── Engram dev bootstrap ─────────────────────────────────"
echo "  repo    : $REPO_ROOT"
echo "  python  : $PYTHON"
if [[ $NO_EXTRAS -eq 1 ]]; then echo "  extras  : (none)"; else echo "  extras  : [all]"; fi
if   [[ $NO_SKILL -eq 1 ]]; then echo "  skill   : (skipped)"
elif [[ $NO_DEV   -eq 1 ]]; then echo "  skill   : copy"
else                              echo "  skill   : symlink (dev)"
fi
if [[ $NO_TEST -eq 1 ]]; then echo "  tests   : (skipped)"; else echo "  tests   : smoke + unit"; fi
echo ""

# ── 1. pip install -e . ─────────────────────────────────────────────────────
if [[ $SKIP_PIP -eq 1 ]]; then
    echo "[1/3] skipped pip install (--skip-pip)"
else
    if [[ $NO_EXTRAS -eq 1 ]]; then
        PIP_TARGET="$REPO_ROOT"
    else
        PIP_TARGET="$REPO_ROOT[all]"
    fi
    echo "[1/3] $PYTHON -m pip install -e \"$PIP_TARGET\""
    "$PYTHON" -m pip install -e "$PIP_TARGET"
    echo "      OK"
fi
echo ""

# ── 2. wire the skill ───────────────────────────────────────────────────────
if [[ $NO_SKILL -eq 1 ]]; then
    echo "[2/3] skipped skill install (--no-skill)"
    echo "      Run later: $PYTHON -m engram.cli install-skill --dev"
else
    skill_args=(-m engram.cli install-skill --force)
    if [[ $NO_DEV -eq 0 ]]; then
        skill_args+=(--dev)
    fi
    echo "[2/3] $PYTHON ${skill_args[*]}"
    "$PYTHON" "${skill_args[@]}"
fi
echo ""

# ── 3. verify with tests ────────────────────────────────────────────────────
if [[ $NO_TEST -eq 1 ]]; then
    echo "[3/3] skipped tests (--no-test)"
    echo "      Run later:"
    echo "      ENGRAM_EMBEDDER=hash $PYTHON tests/smoke_test.py && $PYTHON tests/unit_test.py"
else
    echo "[3/3] ENGRAM_EMBEDDER=hash $PYTHON tests/smoke_test.py"
    ENGRAM_EMBEDDER=hash "$PYTHON" "$REPO_ROOT/tests/smoke_test.py"
    echo ""
    echo "      $PYTHON tests/unit_test.py"
    "$PYTHON" "$REPO_ROOT/tests/unit_test.py"
    echo ""
    echo "      $PYTHON tests/agents_test.py"
    "$PYTHON" "$REPO_ROOT/tests/agents_test.py"
fi
echo ""

echo "Done."
echo ""
echo "Verify CLI is on PATH:"
echo "  $PYTHON -m engram.cli install-skill --check"
echo "  $PYTHON -m engram.cli version"
