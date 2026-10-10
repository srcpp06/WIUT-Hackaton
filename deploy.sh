#!/usr/bin/env bash
# Deploy the pushed HEAD of main to the demo server and record it as a GitHub deployment.
#   ./deploy.sh          code only: the server checks out this commit and rebuilds the container
#   ./deploy.sh --data   also sync models/ (trained artifacts never go through git)
set -euo pipefail
HOST=ubuntu@132.226.186.157
KEY=${KEY:-$HOME/.ssh/oracle_server}
URL=https://qalqon.132-226-186-157.sslip.io
REPO=srcpp06/WIUT-Hackaton
SSH="ssh -i $KEY -o BatchMode=yes"

sha=$(git rev-parse HEAD)
git fetch -q origin
[ "$(git rev-parse origin/main)" = "$sha" ] || { echo "HEAD is not origin/main: push first" >&2; exit 1; }

if [ "${1:-}" = "--data" ]; then
  rsync -az --delete --chmod=F644,D755 -e "$SSH" models/ "$HOST:qalqon/models/"
fi

dep=$(gh api "repos/$REPO/deployments" --jq .id --input - <<JSON
{"ref": "$sha", "environment": "production", "auto_merge": false, "required_contexts": [],
 "description": "Oracle Cloud ARM VM, Docker behind Caddy"}
JSON
)
$SSH "$HOST" "cd ~/qalqon && git fetch -q && git reset -q --hard $sha && docker compose up -d --build"

state=failure
for _ in $(seq 1 60); do
  if curl -fsS "$URL/health" 2>/dev/null | grep -q '"ok"'; then state=success; break; fi
  sleep 3
done
gh api "repos/$REPO/deployments/$dep/statuses" -f state=$state -f environment_url="$URL" -f log_url="$URL/health" > /dev/null
echo "deployed ${sha:0:7}: $state"
[ $state = success ]
