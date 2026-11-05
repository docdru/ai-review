# Changelog

## 1.0.0 – 2026-11-05

First release.

- Composite action; repository-specific prompt, sections, reference docs and
  file lists in `.github/ai-review.toml`, read from the PR's base commit.
- One severity scale P0–P3 on every finding, with per-section labels
  (Critical/High/Medium/Low for security). The verdict is computed from the
  findings; minor findings are folded.
- Context budget derived from the model's context window.
- Oversized PRs get a "review skipped" comment, which replaces an older review.
- Retries for GitHub and Mistral with a job deadline; Mistral timeouts are
  never retried; `length` and `model_length` cut-offs are detected.
- Notices for unreadable changed files and large files without a patch.
- Notebooks reviewed as cell sources; token usage shown in the comment.
