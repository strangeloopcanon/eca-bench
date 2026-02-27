#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_BASE="${REPO_ROOT}/workspaces"

AGENTS=("codex" "gemini")

echo "Setting up clean workspaces..."

for agent in "${AGENTS[@]}"; do
    ws="${WORKSPACE_BASE}/${agent}"

    if [ -d "$ws" ]; then
        echo "  Removing existing workspace: ${agent}/"
        rm -rf "$ws"
    fi

    mkdir -p "$ws"

    cp "${REPO_ROOT}/open_challenge/challenge.py" "$ws/"
    cp "${REPO_ROOT}/open_challenge/verifier.py" "$ws/"
    cp "${REPO_ROOT}/prompt/CHALLENGE.md" "$ws/"

    # Initialize a git repo so Codex doesn't complain
    (cd "$ws" && git init -q && git -c user.name="bench" -c user.email="bench@local" add -A && git -c user.name="bench" -c user.email="bench@local" commit -q -m "initial challenge workspace")

    echo "  Created workspace: ${agent}/"
    echo "    Files: challenge.py, verifier.py, CHALLENGE.md"
done

echo ""
echo "Done. Workspaces ready at: ${WORKSPACE_BASE}/"
echo ""
echo "Next steps:"
echo "  ./scripts/run_codex.sh"
echo "  ./scripts/run_gemini.sh"
