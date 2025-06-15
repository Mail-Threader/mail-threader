# Enron Email Analysis Pipeline

This project provides a comprehensive pipeline for analyzing the Enron email dataset. It includes modules for data preparation, summarization/classification, visualization, and story development, along with a modern web interface for interacting with the analysis results.

## Project Structure

- `src/`: Contains all backend source code
  - `main.py`: Main entry point for the pipeline
  - `data_preparation/`: Module for loading and preprocessing email data
    - `data_preparation.py`: Implementation of the DataPreparation class
    - `__init__.py`: Exports the DataPreparation class
  - `summarization_classification/`: Module for topic modeling, clustering, and analysis
    - `summarization_classification.py`: Implementation of the SummarizationClassification class
    - `__init__.py`: Exports the SummarizationClassification class
  - `visualization/`: Module for creating visualizations
    - `visualization.py`: Implementation of the Visualization class
    - `__init__.py`: Exports the Visualization class
  - `story_development/`: Module for generating narratives from the data
    - `story_development.py`: Implementation of the StoryDevelopment class
    - `__init__.py`: Exports the StoryDevelopment class
  - `utils/`: Utility functions used across modules
    - `utils.py`: Implementation of utility functions
    - `__init__.py`: Exports utility functions
  - `__init__.py`: Package initialization file
- `frontend/`: Next.js web application
  - `app/`: Next.js app directory with pages and layouts
  - `components/`: Reusable React components
  - `lib/`: Frontend utility functions and API clients
  - `db/`: Database models and utilities
- `tests/`: Test files
  - `unit/`: Unit tests for individual components
  - `integration/`: Integration tests for module interactions
- `data/`: Directory containing the Enron email dataset
- `output/`: Directory where all results will be stored (created automatically)
- `.github/`: GitHub Actions workflows and templates
- `.vscode/`: VS Code configuration files

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- Required Python packages (listed in requirements.txt)
- Required Node.js packages (listed in frontend/package.json)

## Quick Start for New Team Members

If you're new to this project, we've created resources to help you get started quickly:

1. Read the [Getting Started Guide](GETTING_STARTED.md) for comprehensive instructions
2. Run the automated setup script:

   **For Linux/macOS users:**

   ```bash
   chmod +x setup_dev_environment.sh
   ./setup_dev_environment.sh
   ```

   **For Windows users:**

   ```cmd
   setup_dev_environment.bat
   ```

These scripts will set up your development environment automatically, including creating a virtual environment, installing dependencies, and setting up pre-commit hooks.

## Installation

### For Users

1. Clone this repository
2. Install the required dependencies:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
```

### For Developers

1. Clone this repository
2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode with all development dependencies:

```bash
pip install -e ".[dev]"
```

4. Install frontend dependencies:

```bash
cd frontend
npm install
```

5. Install pre-commit hooks:

```bash
pre-commit install
```

## How to Run

### Backend

The main entry point for the pipeline is `src/main.py`. You can run it with various command-line arguments to customize the execution.

#### Basic Usage

```bash
python src/main.py
```

This will run the entire pipeline with default settings, using the data in the `data/` directory and storing results in the `output/` directory.

#### Command-line Arguments

- `--data-dir`: Directory containing the email data files (default: `./data/`)
- `--output-dir`: Directory to store all output files (default: `./output/`)
- `--skip`: Steps to skip (can specify multiple steps)
  - Available steps: `data-prep`, `analysis`, `vis`, `story`
  - Example: `--skip data-prep analysis`
- `--run`: Steps to run (can specify multiple steps)
  - Available steps: `data-prep`, `analysis`, `vis`, `story`
  - Example: `--run vis story`

### Examples

Run only the visualization and story development steps:

```bash
python src/main.py --run vis story
```

Skip data preparation and analysis steps:

```bash
python src/main.py --skip data-prep analysis
```

Use a different data directory:

```bash
python src/main.py --data-dir /path/to/enron/emails
```

### Frontend

To run the frontend development server:

```bash
cd frontend
npm run dev
```

This will start the Next.js development server at http://localhost:3000.

## Development

### Using the Makefile

The project includes a Makefile with several useful commands:

```bash
make help              # Display help information
make clean            # Remove build artifacts
make lint             # Check code style
make format           # Format code
make test             # Run tests
make coverage         # Generate coverage reports
make docs             # Generate documentation
make dist             # Build package
make install          # Install package
make dev-install      # Install in development mode
```

### Code Style

This project uses the following tools to ensure code quality:

- **Backend**:
  - Black: Code formatting
  - isort: Import sorting
  - Ruff: Linting
  - mypy: Type checking

- **Frontend**:
  - ESLint: JavaScript/TypeScript linting
  - Prettier: Code formatting
  - TypeScript: Static type checking

### Testing

#### Backend Tests

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=src --cov-report=html
```

#### Frontend Tests

```bash
cd frontend
npm test
```

### Documentation

We use Sphinx for backend documentation and Next.js documentation for the frontend.

To build the backend documentation:

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`.

## Contributing

Please see the [Contributing Guide](docs/source/contributing.rst) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
