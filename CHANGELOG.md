# Changelog

## 1.2.0 – 2026-11-20

- **Related context (`[related]`, off by default).** Until now the reviewer saw
  the diff, the changed files and the reference docs, and nothing else of the
  repository: not the callers of a changed function, not the test next to a
  changed file. Four additions, each switched on in the config:
  - `tests = true`: the tests that belong to a changed file by name
    (`x.test.ts`, `x.spec.ts`, `test_x.py`, `x_test.go`, `XTest.kt`).
  - `imports = true`: one step along the imports, in both directions: files
    that import a changed file and files it imports. JavaScript, TypeScript and
    Python; relative imports, and absolute Python modules that fit one file.
  - `nearby = true`: the names of the files and folders next to the changed
    files, so the reviewer knows what exists.
  - `[[related.rules]]`: "when a path like this changes, also send these
    files", for contracts and interfaces a change must honour.
- Related files only use room the PR left free (at most `related.max_chars`,
  200,000 by default), never push out the diff or the changed files, and are
  capped per file and in number. What did not fit is named in the prompt.
- The comment's last line says what was sent ("Related context: 2 tests,
  3 of 9 importing files"), and `--dry-run` shows the part and its size.
- For tests, imports and nearby names the repository is downloaded once as an
  archive and read in memory: nothing is written to disk, nothing is run,
  links are ignored. If the download fails, the review still runs, with a
  notice. Rules alone need no download.
- Without a `[related]` table nothing changes: same prompt, same API calls.

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
- **A changed reference doc is sent once.** When a PR changes a reference doc
  (say the spec), its full new version was sent twice: as a reference doc and
  again as a changed file, where it competed with the code for room and was
  the first thing trimmed. Now the diff keeps its patch, and the changed-files
  list points to the reference copy. A doc longer than its `max_chars` still
  keeps its full copy as a changed file. On kidsmedia this saves ~95–115k
  characters on every PR that touches its 114k-character spec.

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
