# Data Preparation — Improvement Plan

## 1. Thread Reconstruction

**Goal**: Build full email thread trees from the flat list of extracted emails.

**How**:
- `References` and `In-Reply-To` headers already parsed into `parent_message_id` — use them to link emails
- For emails without headers (forwarded/original blocks), infer parent from `main_id`
- Build a thread map: `{thread_root_id: [email_1, email_2, ...]}` sorted by date
- Add a `thread_id` column and `thread_depth` (0 for root, 1, 2, ...) to the output DataFrame

**Value**: Story development needs thread context to build narratives. Currently threads are implicit.

**Effort**: ~2 hours. No new dependencies. Pure pandas.

---

## 2. Folder / Origin Metadata

**Goal**: Preserve the PST folder structure (Inbox, Sent, Deleted, etc.) that's currently lost.

**How**:
- The input directory structure encodes this: `data/albert/Inbox/` → folder = `Inbox`, custodian = `albert`
- During file walk, extract `custodian` (parent folder name) and `folder` (subfolder name like `Inbox`, `Sent_Items`, `Deleted_Items`)
- Add `custodian` and `folder` columns to the DataFrame
- Use `_Sent` suffix detection to identify "Sent" folder emails (already partially present as a type hint)

**Value**: Enables downstream filtering (e.g., "only inbox emails"), and story development can distinguish sent vs received.

**Effort**: ~1 hour. No new dependencies.

---

## 3. Content-Type Classifier

**Goal**: Tag emails with a content type so downstream stages can filter or handle differently.

**Types**:
| Type | Heuristic |
|---|---|
| `conversation` | Has `In-Reply-To` or `References`, or subject starts with `Re:` |
| `newsletter` | High link-to-text ratio, unsubscribe links, no personal greeting |
| `log` | No `From:` domain matches Enron, system-generated style (e.g. `8816. 2`) |
| `attachment_notice` | Body contains "ATTENTION", "attachment", "file attached" |
| `unknown` | Falls through all heuristics |

**Value**: The `8816. 2` log file and newsletter noise get flagged; story development can focus on conversations.

**Effort**: ~3 hours for heuristics. Optional ML version would be more.

---

## 4. Fuzzy Duplicate Detection

**Goal**: Catch near-duplicate emails that exact-match dedup misses.

**How**:
- After exact dedup, compute TF-IDF on `body` + `subject`
- Use cosine similarity with a threshold (e.g., 0.95) to group near-duplicates
- Keep the longest version (most complete), mark duplicates in a `duplicate_of` column

**Value**: The 45% dedup rate is from exact matches. There are likely more near-duplicates (same email forwarded multiple times with slightly different headers).

**Effort**: ~3 hours. Requires `scikit-learn` (already installed).

---

## 5. Multiprocessing

**Goal**: Reduce the 103s runtime.

**How**:
- `process_all_emails` iterates 7220 files sequentially — embarrassingly parallel
- Use `concurrent.futures.ProcessPoolExecutor` to process files in chunks
- Merge partial DataFrames and dedup at the end
- Number of workers = `os.cpu_count()` (typically 8–16 on modern hardware)

**Expected gain**: 103s → ~20–30s depending on I/O and CPU count.

**Effort**: ~2 hours. Uses stdlib `concurrent.futures`.

---

## 6. Validation Suite

**Goal**: Unit tests for each module to prevent regressions.

**Tests**:

| Module | Tests |
|---|---|
| `utils.py` | `normalize_dates` with various formats, `strip_x500_dn` |
| `parser.py` | `parse_headers_main` (folded headers, MIME), `parse_addresses`, `decode_mime_header`, `extract_fields_line_by_line` |
| `cleaner.py` | `clean_body` (QP decode, HTML stripping, whitespace normalization) |
| `extractor.py` | `split_all_messages` (forwarded + original ordering), `extract_all_emails` (full round-trip) |
| `pipeline.py` | `process_all_emails` with `limit`, dedup behavior |

Use `pytest` (already in dependencies). Keep test data in `tests/fixtures/`.

**Value**: Confidence to refactor without breakage. Currently no tests exist.

**Effort**: ~4 hours for comprehensive coverage.

---

## Priority Recommendation

| # | Item | Value | Effort | Quick Win |
|---|---|---|---|---|
| 1 | Thread reconstruction | High | 2h | ✅ Yes |
| 2 | Folder metadata | Medium | 1h | ✅ Yes |
| 6 | Validation suite | High | 4h | ❌ No |
| 3 | Content-type classifier | Medium | 3h | ❌ No |
| 5 | Multiprocessing | Low | 2h | ❌ No |
| 4 | Fuzzy dedup | Medium | 3h | ❌ No |

**Suggested order**: Threads → Folder metadata → Tests → Classifier → Multiprocessing → Fuzzy dedup
