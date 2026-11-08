# ai-review

Automated AI review of pull requests with the Mistral API, as a GitHub
composite action. One implementation for all repositories; each repository
describes what matters to it in `.github/ai-review.toml`.

The review is posted as **one PR comment**, updated on every push. It is
advisory: not a required check, and its findings should be verified.

## Why

- **One reviewer for many repositories.** The review logic, its safety
  measures and its tests live in one place. A fix or improvement reaches every
  repository as a normal Dependabot PR that bumps the pinned version, so each
  upgrade is itself reviewed.
- **Specialized where it matters.** Each repository describes its project,
  its review checklist and its reference documents (spec, README, project
  rules) in a small config file. The rest is shared.
- **Comparable results.** Every finding carries the same P0–P3 severity, and
  the verdict follows from the findings, so reviews read the same way across
  repositories.
- **Safe by construction.** The action never checks out or runs PR code, its
  instructions come from the base branch, and text that tries to steer the
  review is reported instead of followed.
- **No silent gaps.** Truncated context, missing documents and skipped PRs
  are stated in the comment; errors fail the run.

## What it does

1. Reads `.github/ai-review.toml` from the PR's **base** commit (`main`), so a
   PR can't rewrite its own review instructions. A config that is new in the
   PR (first setup) is taken from the PR, with a notice.
2. Gathers, at the PR's **head** commit and through the GitHub API as text
   only: the diff with one header per file, the full content of every changed
   file, the PR title and description, and the configured reference documents
   (spec, README, AGENTS.md, ...). Notebooks are sent as cell sources without
   outputs. Generated and binary files are listed by name only.
3. Sends it to Mistral and asks for findings per configured section, plus a
   built-in "Surprising changes" section.
4. Posts or updates the comment: findings sorted by severity, minor ones
   folded away, a verdict computed from the findings, context notices, the
   model and token usage.

Nothing from the PR is checked out, installed, imported or executed. The
script uses only the Python standard library; nothing is installed.

## Severity scale

The same four levels everywhere. The model returns the level; the label
depends on the section.

| Level | Security sections | Other sections | Verdict |
|---|---|---|---|
| **P0** | Critical | Blocker: breaks a rule, security or data | 🔴 `request_changes` |
| **P1** | High | Bug: likely bug or spec/doc mismatch | 🟠 `comment` |
| **P2** | Medium | Maintainability | 🟢 `approve` |
| **P3** | Low | Minor, optional (folded in the comment) | 🟢 `approve` |

Style nits that linters cover are never reported.

**Evidence:** every finding about a file quotes the line it is about. The
quote is checked against the PR and shown under the finding, so a finding can
be verified at a glance. A P0 whose quote is missing or not found in the PR is
shown as P1 (marked "was P0"), so one invented finding can't block a PR.

## Setup in a repository

1. Add the repository secret `MISTRAL_API_KEY`
   (*Settings → Secrets and variables → Actions*).
2. Copy [`examples/ai-review.yml`](examples/ai-review.yml) to
   `.github/workflows/ai-review.yml` and pin `docdru/ai-review` to a release
   commit SHA. Keep the `github-actions` ecosystem in `dependabot.yml`, so
   upgrades arrive as PRs.
3. Copy [`examples/ai-review.toml`](examples/ai-review.toml) to
   `.github/ai-review.toml` and describe the project.

Runs for non-draft PRs from branches in the repository by authors with write
access; fork, bot and draft PRs are skipped. Mark a draft "Ready for review"
to trigger it.

### Action inputs

| Input | Default | |
|---|---|---|
| `mistral-api-key` | – (required) | pass `${{ secrets.MISTRAL_API_KEY }}` |
| `github-token` | `${{ github.token }}` | needs `contents: read`, `pull-requests: write` |
| `config` | `.github/ai-review.toml` | path in the calling repository |

## Configuration

`.github/ai-review.toml` (see the [annotated example](examples/ai-review.toml)).
Unknown keys are errors, so typos fail loudly.

| Key | Default | Meaning |
|---|---|---|
| `project` | – (required) | what the repository is: "You are a senior reviewer for …" |
| `instructions` | `""` | extra rules, known tricky areas |
| `model` | `mistral-large-2512` | pinned Mistral model |
| `context_tokens` | known per model | context window; required for models the action doesn't know |
| `max_output_tokens` | `8192` | room for the reply |
| `reference_docs` | `[]` | `{ path, role, required = true, max_chars = 100000 }` |
| `generated_files` / `generated_suffixes` | `[]` | not sent, only named |
| `extra_binary_suffixes` | `[]` | added to the built-in list |
| `max_files` | `50` | more changed (non-binary) files: the PR is skipped with a comment |
| `max_file_chars` | `150000` | per changed file |
| `chars_per_token` | `3.0` | sizes the context budget (1.5–4.5); lower it if the model reports an oversized request |
| `[[sections]]` | – (≥ 1) | `key`, `title`, `checklist`, `security = false`, `reference = false`, `labels` (4 strings) |

### Context budget

The budget is derived from the model: `(context_tokens − max_output_tokens) ×
chars_per_token` characters, ≈ 743k for Mistral Large 3. Real reviews measured
about 3.6 characters per token; the default of 3 leaves headroom for text that
tokenizes worse.

New files are sent once: the diff only names them, and their full content
follows with the other changed files. A changed reference doc is sent once too:
its patch in the diff, its full new version with the reference docs. Ordinary
PRs are far below the budget and see no truncation. Above it, changed files are trimmed first (spread
proportionally), then the diff; reference docs never. Every cut is marked for
the model and listed as a notice in the comment. PRs with more than
`max_files` non-binary files or a diff over 1.5M characters get a "review
skipped" comment instead, which also replaces an older, now stale review.

### Failures

Errors fail the run with an `::error::` annotation: missing or revoked
`MISTRAL_API_KEY`, missing permissions, an invalid config, a cut-off reply.
GitHub calls are retried on 429/5xx/network errors (2/4/8 s, honouring
`Retry-After`); Mistral gets one retry on 429/5xx, never after a timeout (the
request may already have been billed), and no retry that can't finish inside
the job's 15 minutes.

## Try it locally

Builds the prompt and prints its size per part; calls nothing, posts nothing:

```sh
GITHUB_TOKEN=$(gh auth token) python3 ai_review.py --repo OWNER/REPO --pr 42 --dry-run
# test a config change before merging it, and look at the whole prompt:
GITHUB_TOKEN=$(gh auth token) python3 ai_review.py --repo OWNER/REPO --pr 42 \
    --dry-run --config path/to/ai-review.toml --dump /tmp/prompt.txt
```

## Development

```sh
python3 -m unittest discover -s tests -v   # standard library, no network
ruff check . && ruff format --check . && mypy
```

Releases: bump `__version__` in `ai_review.py` and `pyproject.toml`, add a
`CHANGELOG.md` entry, merge, tag `vX.Y.Z`. Callers pin the tag's commit SHA.
