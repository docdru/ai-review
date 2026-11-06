# Changelog

## 1.1.0 – 2026-11-06

- **Evidence for findings.** Every finding about a file carries a `quote`: one
  line copied from the diff or the file. The script checks it against the
  context the model saw (whitespace-insensitive, diff markers ignored) and
  shows it under the finding. A **P0 whose quote is missing or not found is
  shown as P1**, so a single invented finding can no longer turn the verdict
  into `request_changes`.
- Prompt: re-read the quoted line before reporting and drop findings the code
  already handles; cite only what the context contains (no invented URLs or
  document sections); a P0 security finding must name how it can be exploited.
- **More room for large PRs.** New files are sent once: the diff only names
  them, since their full content is sent anyway (it used to be sent twice, as
  an all-"+" patch and as the file). The budget assumes 3 characters per token
  instead of 2, based on ~3.6 measured in real reviews: ≈ 743k instead of
  ≈ 496k characters for Mistral Large 3. Configurable as `chars_per_token`.
- Binary files no longer count toward `max_files`.

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
