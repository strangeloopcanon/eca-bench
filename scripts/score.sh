#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_BASE="${REPO_ROOT}/workspaces"

WIDTH="${1:-32}"
HORIZON="${2:-2}"
TEST_EXAMPLES="${3:-10000}"
SEED="${4:-1337}"

echo "=== ECA-Bench Scoreboard ==="
echo "Task: width=${WIDTH}, horizon=${HORIZON}, test_examples=${TEST_EXAMPLES}, seed=${SEED}"
echo ""

printf "%-12s %-10s %-12s %-10s %s\n" "AGENT" "ACCURACY" "PARAMS" "PASS?" "NOTES"
printf "%-12s %-10s %-12s %-10s %s\n" "-----" "--------" "------" "-----" "-----"

for agent_dir in "${WORKSPACE_BASE}"/*/; do
    agent="$(basename "$agent_dir")"
    submission="${agent_dir}submission.py"

    if [ ! -f "$submission" ]; then
        printf "%-12s %-10s %-12s %-10s %s\n" "$agent" "-" "-" "NO" "no submission.py found"
        continue
    fi

    score_file="${agent_dir}score.json"

    result=$(cd "$agent_dir" && python verifier.py \
        --submission submission.py \
        --width "$WIDTH" \
        --horizon "$HORIZON" \
        --test-examples "$TEST_EXAMPLES" \
        --seed "$SEED" \
        --json-out score.json 2>&1) || true

    if [ -f "$score_file" ]; then
        accuracy=$(python3 -c "import json; d=json.load(open('${score_file}')); print(f\"{d.get('exact_match', 0):.4f}\")")
        params=$(python3 -c "import json; d=json.load(open('${score_file}')); print(d.get('trainable_params', '?'))")
        passed=$(python3 -c "import json; d=json.load(open('${score_file}')); print('YES' if d.get('target_met') else 'NO')")
        error=$(python3 -c "import json; d=json.load(open('${score_file}')); print(d.get('error', ''))")

        notes=""
        if [ -n "$error" ]; then
            notes="error: ${error}"
        fi

        printf "%-12s %-10s %-12s %-10s %s\n" "$agent" "$accuracy" "$params" "$passed" "$notes"
    else
        printf "%-12s %-10s %-12s %-10s %s\n" "$agent" "-" "-" "ERROR" "verifier failed to produce score"
    fi
done

echo ""

# Check for reports
echo "=== Reports ==="
for agent_dir in "${WORKSPACE_BASE}"/*/; do
    agent="$(basename "$agent_dir")"
    report="${agent_dir}REPORT.md"
    if [ -f "$report" ]; then
        lines=$(wc -l < "$report" | tr -d ' ')
        echo "  ${agent}: REPORT.md (${lines} lines)"
    else
        echo "  ${agent}: no REPORT.md"
    fi
done
