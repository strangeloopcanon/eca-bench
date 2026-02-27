#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="${REPO_ROOT}/workspaces/gemini"
PROMPT="${WORKSPACE}/CHALLENGE.md"

if [ ! -d "$WORKSPACE" ]; then
    echo "Error: workspace not found. Run ./scripts/setup.sh first."
    exit 1
fi

echo "Starting Gemini agent in: ${WORKSPACE}"
echo "Prompt: ${PROMPT}"
echo ""

cd "$WORKSPACE"
gemini -p "$(cat "$PROMPT")" --approval-mode yolo

echo ""
echo "Gemini finished. Workspace: ${WORKSPACE}"
echo "Run ./scripts/score.sh to evaluate results."
