#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="${REPO_ROOT}/workspaces/codex"
PROMPT="${WORKSPACE}/CHALLENGE.md"
TRANSCRIPT="${WORKSPACE}/codex_transcript.md"

if [ ! -d "$WORKSPACE" ]; then
    echo "Error: workspace not found. Run ./scripts/setup.sh first."
    exit 1
fi

echo "Starting Codex agent in: ${WORKSPACE}"
echo "Prompt: ${PROMPT}"
echo "Transcript will be saved to: ${TRANSCRIPT}"
echo ""

codex exec \
    --full-auto \
    -C "$WORKSPACE" \
    --skip-git-repo-check \
    -o "$TRANSCRIPT" \
    "$(cat "$PROMPT")"

echo ""
echo "Codex finished. Workspace: ${WORKSPACE}"
echo "Run ./scripts/score.sh to evaluate results."
