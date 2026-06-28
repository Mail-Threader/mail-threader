# Enron Email Analysis Pipeline

A comprehensive pipeline for analyzing the Enron email dataset — from raw text extraction to story development and visualization. Includes a modular Python backend and a Next.js interactive dashboard.

## Project Structure

```
src/
  data_preparation/        Email extraction, cleaning, threading, dedup
    pipeline.py            Orchestrator (parallel processing)
    parser.py              Header parsing, MIME decode, address extraction
    cleaner.py             Body cleaning (QP, HTML, quoting)
    extractor.py           Message split (main/forwarded/original)
    utils.py               Date normalization, X500 DN stripping
    dedup.py               TF-IDF fuzzy duplicate detection
    threading.py           BFS thread tree reconstruction
    classifier.py          Heuristic content-type tagging
  summarization_classification/
  visualization/
  story_development/
  database/
  utils/
  main.py                  CLI entry point

frontend/                  Next.js web application
  app/                     Pages and layouts
  components/              React components (shadcn/ui)
  db/                      Database models

tests/                     Test suite (65 tests)
  test_parser.py
  test_cleaner.py
  test_extractor.py
  test_pipeline.py
  test_utils.py

data/                      Raw Enron .txt files (gitignored)
output/                    Generated results (gitignored)
```

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Node.js 18+ (for frontend)

## Quick Start

```bash
# Clone and enter
git clone https://github.com/Mail-Threader/mail-threader.git
cd mail-threader

# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -e ".[dev]"

# Run the full pipeline
uv run python -c "
from data_preparation import DataPreparation
dp = DataPreparation()
df = dp.process_all_emails()
dp.save_to_pickle(df)
dp.save_to_json(df)
"

# Run tests
uv run pytest tests/ -v --tb=short
```

> **Note**: `uv` is the recommended package manager. If using pip, replace `uv pip` with `pip` and `uv run` with `python`.

## Data Preparation Pipeline

The data preparation module transforms 7,220 raw Enron `.txt` files into a structured 20-column DataFrame in ~33 seconds (22-core parallel).

### Pipeline steps

| Step | Module | Description |
|---|---|---|
| File discovery | `pipeline.py` | Walks `./data/`, collects 7,220 files |
| Parallel extraction | `extractor.py` | Splits files into main/forwarded/original blocks, parses headers, cleans bodies |
| Exact dedup | `pipeline.py` | `drop_duplicates` on [date, from, to, subject, body] → 12,111→6,604 |
| Fuzzy dedup | `dedup.py` | TF-IDF + NearestNeighbors (0.90 threshold) → 733 more dups marked |
| Threading | `threading.py` | BFS from parent_message_id → 4,295 threads, up to depth 1 |
| Classification | `classifier.py` | Heuristic → conversation / attachment_notice / newsletter / log |

### Output columns (20)

`message_id`, `parent_message_id`, `main_id`, `filename`, `type`, `date`, `from`, `to`, `cc`, `X-From`, `X-To`, `X-cc`, `subject`, `body`, `has_body`, `is_html`, `duplicate_of`, `thread_id`, `thread_depth`, `content_type`

### Key improvements (data-prep-v2)

- **5-module refactor**: Monolithic `data_preparation.py` → `parser.py`, `cleaner.py`, `extractor.py`, `utils.py`, `pipeline.py`
- **Thread reconstruction**: BFS thread trees via `parent_message_id`
- **Content-type classifier**: Heuristic tagging (85% conversation, 12% attachment, 2% newsletter)
- **Fuzzy dedup**: TF-IDF cosine similarity at 0.90 threshold
- **Multiprocessing**: ~88s → ~33s (2.7x speedup)
- **Validation suite**: 65 pytest tests covering all modules

## CLI Usage

```bash
# Run specific steps
uv run python src/main.py --run vis story

# Skip steps
uv run python src/main.py --skip data-prep analysis

# Custom data directory
uv run python src/main.py --data-dir /path/to/emails
```

Available steps: `data-prep`, `analysis`, `vis`, `story`

## Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

## Development

### Commands

```bash
uv run pytest            # Run tests
uv run pytest --cov=src  # With coverage
uv run pytest tests/ -v  # Verbose
```

### Code style

- Backend: yapf (tab indentation)
- Frontend: ESLint + Prettier + TypeScript

### Documentation

```bash
cd docs && make html     # Build Sphinx docs → docs/build/html/
```

Full module reference: [`data_preparation.md`](data_preparation.md)

## Project status

[data-prep-v2](https://github.com/Mail-Threader/mail-threader/tree/data-prep-v2) contains reworked extraction with threading, classification, dedup, multiprocessing. See branch for full commit history.
