#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: scripts/start_parallel_task.sh [--mode worktree|current] [--skip-pr-check] [--thread-id id] [--thread-name name] [--routing-workflow workflow] [--routing-confidence score] [--routing-source source] [--routing-strategy strategy] [--source-prompt text|--source-prompt-file path] <workflow> <topic> [issue-number]" >&2
}

mode="worktree"
skip_pr_check="false"
thread_id="${CODEX_THREAD_ID:-}"
thread_name=""
routing_workflow=""
routing_confidence=""
routing_source=""
routing_strategy="manual"
source_prompt=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="${2:-}"
      shift 2
      ;;
    --skip-pr-check)
      skip_pr_check="true"
      shift
      ;;
    --thread-id)
      thread_id="${2:-}"
      shift 2
      ;;
    --thread-name)
      thread_name="${2:-}"
      shift 2
      ;;
    --routing-workflow)
      routing_workflow="${2:-}"
      shift 2
      ;;
    --routing-confidence)
      routing_confidence="${2:-}"
      shift 2
      ;;
    --routing-source)
      routing_source="${2:-}"
      shift 2
      ;;
    --routing-strategy)
      routing_strategy="${2:-}"
      shift 2
      ;;
    --source-prompt)
      source_prompt="${2:-}"
      shift 2
      ;;
    --source-prompt-file)
      source_prompt="$(<"${2:-}")"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 1
fi

workflow="$1"
topic="$2"
issue_number="${3:-}"

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

topic_slug="$(printf '%s' "$topic" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
if [[ -z "$topic_slug" ]]; then
  echo "Topic must contain at least one alphanumeric character." >&2
  exit 1
fi

if [[ -n "$issue_number" && ! "$issue_number" =~ ^[0-9]+$ ]]; then
  echo "Issue number must be numeric when provided." >&2
  exit 1
fi

if [[ "$mode" != "worktree" && "$mode" != "current" ]]; then
  echo "Mode must be either 'worktree' or 'current'." >&2
  exit 1
fi

branch="codex/$workflow/$topic_slug"
base_branch="dev/$workflow"
if [[ "$workflow" == "main" ]]; then
  base_branch="main"
fi
worktree_path="$(cd .. && pwd)/wt-$workflow-$topic_slug"
branch_slug="${branch//\//-}"
workspace_root="$worktree_path"
if [[ "$mode" == "current" ]]; then
  workspace_root="$repo_root"
fi
routing_workflow="${routing_workflow:-$workflow}"
routing_source="${routing_source:-$routing_strategy}"
pr_check_mode="gh"
last_thread_seen_at=""
current_branch="$(git branch --show-current 2>/dev/null || true)"
adopt_existing_current="false"
if [[ -n "$thread_id" ]]; then
  last_thread_seen_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
fi

python3 scripts/check_branch_ownership.py --branch "$branch" --base "$base_branch" --changed-file /dev/null

if [[ "$skip_pr_check" == "true" ]]; then
  pr_check_mode="skipped"
elif ! command -v gh >/dev/null 2>&1; then
  pr_check_mode="mcp-required"
  echo "GitHub CLI (gh) is unavailable. Skipping local duplicate PR check; use GitHub MCP search before opening the task PR." >&2
elif ! gh auth status >/dev/null 2>&1; then
  pr_check_mode="mcp-required"
  echo "GitHub CLI is not authenticated. Skipping local duplicate PR check; use GitHub MCP search before opening the task PR." >&2
fi

git fetch origin

if ! git show-ref --verify --quiet "refs/heads/$base_branch"; then
  if git show-ref --verify --quiet "refs/remotes/origin/$base_branch"; then
    git branch --track "$base_branch" "origin/$base_branch" >/dev/null 2>&1 || true
  fi
fi

if ! git show-ref --verify --quiet "refs/heads/$base_branch"; then
  git fetch origin "$base_branch:$base_branch" >/dev/null 2>&1 || true
fi

if ! git show-ref --verify --quiet "refs/heads/$base_branch"; then
  echo "Local tracking branch '$base_branch' is missing and could not be created from origin/$base_branch." >&2
  exit 1
fi

if [[ "$mode" == "current" && "$current_branch" == "$branch" ]]; then
  adopt_existing_current="true"
fi

if git show-ref --verify --quiet "refs/heads/$branch"; then
  if [[ "$adopt_existing_current" != "true" ]]; then
    echo "Local branch '$branch' already exists. Reuse or finish that task before starting another one with the same slug." >&2
    exit 1
  fi
fi

if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
  if [[ "$adopt_existing_current" != "true" ]]; then
    echo "Remote branch '$branch' already exists. Pick a different topic slug or finish the existing task first." >&2
    exit 1
  fi
fi

if [[ "$mode" == "worktree" && -e "$worktree_path" ]]; then
  echo "Worktree path '$worktree_path' already exists." >&2
  exit 1
fi

if [[ "$pr_check_mode" == "gh" && "$adopt_existing_current" != "true" ]]; then
  existing_prs="$(gh pr list --state open --head "$branch" --json number,url --jq 'length')"
  if [[ "$existing_prs" != "0" ]]; then
    echo "An open pull request already exists for '$branch'. Refusing to create a duplicate task workspace." >&2
    exit 1
  fi
fi

if [[ "$mode" == "worktree" ]]; then
  git worktree add "$worktree_path" -b "$branch" "$base_branch"
else
  if [[ "$adopt_existing_current" == "true" ]]; then
    git checkout "$branch" >/dev/null 2>&1 || true
  else
    if [[ -n "$(git status --short)" ]]; then
      echo "Current workspace has uncommitted changes. Use a clean cloud workspace before routing into current mode." >&2
      exit 1
    fi
    git checkout -b "$branch" "$base_branch"
  fi
fi

python3 -m venv "$workspace_root/.venv"
mkdir -p "$workspace_root/.task-runtime/tmp" "$workspace_root/outputs/$topic_slug"
cat >"$workspace_root/.task-runtime/task.env" <<EOF
export LARRICK_TASK_WORKFLOW="$workflow"
export LARRICK_TASK_BRANCH="$branch"
export LARRICK_TASK_BASE_BRANCH="$base_branch"
export LARRICK_TASK_TOPIC="$topic_slug"
export LARRICK_TASK_TMPDIR="$workspace_root/.task-runtime/tmp"
export LARRICK_TASK_OUTPUT_DIR="$workspace_root/outputs/$topic_slug"
export LARRICK_DOCKER_NAMESPACE="$branch_slug"
export LARRICK_TASK_THREAD_ID="$thread_id"
export LARRICK_TASK_THREAD_NAME="$thread_name"
export LARRICK_TASK_ROUTING_WORKFLOW="$routing_workflow"
export LARRICK_TASK_ROUTING_CONFIDENCE="$routing_confidence"
export LARRICK_TASK_ROUTING_SOURCE="$routing_source"
export LARRICK_TASK_ROUTING_STRATEGY="$routing_strategy"
export LARRICK_TASK_WORKSPACE_MODE="$mode"
export LARRICK_TASK_PR_CHECK_MODE="$pr_check_mode"
export LARRICK_TASK_CONVERSATION_STATE="active"
export LARRICK_TASK_ARCHIVE_ELIGIBLE="false"
export LARRICK_TASK_LAST_THREAD_SEEN_AT="$last_thread_seen_at"
EOF

task_context_args=(
  task-context
  --workflow "$workflow"
  --topic "$topic_slug"
  --worktree "$workspace_root"
  --branch "$branch"
  --docker-namespace "$branch_slug"
  --workspace-mode "$mode"
)
if [[ -n "$issue_number" ]]; then
  task_context_args+=(--issue-number "$issue_number")
fi
if [[ -n "$thread_id" ]]; then
  task_context_args+=(--thread-id "$thread_id")
fi
if [[ -n "$thread_name" ]]; then
  task_context_args+=(--thread-name "$thread_name")
fi
if [[ -n "$routing_workflow" ]]; then
  task_context_args+=(--routing-workflow "$routing_workflow")
fi
if [[ -n "$routing_confidence" ]]; then
  task_context_args+=(--routing-confidence "$routing_confidence")
fi
if [[ -n "$routing_source" ]]; then
  task_context_args+=(--routing-source "$routing_source")
fi
if [[ -n "$routing_strategy" ]]; then
  task_context_args+=(--routing-strategy "$routing_strategy")
fi
if [[ -n "$source_prompt" ]]; then
  task_context_args+=(--source-prompt "$source_prompt")
fi
if [[ -n "$last_thread_seen_at" ]]; then
  task_context_args+=(--last-thread-seen-at "$last_thread_seen_at")
fi
python3 scripts/plan_github_concierge.py "${task_context_args[@]}" \
  >"$workspace_root/.task-runtime/task.json"

printf 'Workspace mode: %s\n' "$mode"
printf 'Workspace root: %s\n' "$workspace_root"
printf 'Branch: %s\n' "$branch"
printf 'Base branch: %s\n' "$base_branch"
printf 'Adopted existing current branch: %s\n' "$adopt_existing_current"
printf 'PR duplicate check: %s\n' "$pr_check_mode"
printf 'Runtime env: %s\n' "$workspace_root/.task-runtime/task.env"
printf 'Task context: %s\n' "$workspace_root/.task-runtime/task.json"
