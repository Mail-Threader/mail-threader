# Getting Started with the Enron Email Analysis Pipeline

This guide will help new team members get up and running with the project quickly. It provides step-by-step instructions for setting up your development environment, understanding the project structure, and contributing to the codebase.

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Understanding the Project Structure](#understanding-the-project-structure)
3. [Running the Application](#running-the-application)
4. [Development Workflow](#development-workflow)
5. [Using the Makefile](#using-the-makefile)
6. [Debugging](#debugging)
7. [Contributing Guidelines](#contributing-guidelines)

## Initial Setup

### Prerequisites

Before you begin, make sure you have the following installed:
- Python 3.9 or higher
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/siddhantdalvi3/HMI-project.git
cd HMI-project
```

### Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to isolate the project dependencies:

```bash
python -m venv .venv
```

Activate the virtual environment:

- On Windows:
  ```bash
  .venv\Scripts\activate
  ```

- On macOS/Linux:
  ```bash
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies

For regular users:
```bash
pip install -r requirements.txt
```

For developers (includes development tools):
```bash
pip install -e ".[dev]"
```

### Step 4: Install Pre-commit Hooks

Pre-commit hooks help ensure code quality before committing changes:

```bash
pre-commit install
```

## Understanding the Project Structure

The project is organized as follows:

- `src/`: Contains all source code
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
- `tests/`: Test files
  - `unit/`: Unit tests for individual components
  - `integration/`: Integration tests for module interactions
- `data/`: Directory containing the Enron email dataset
- `output/`: Directory where all results will be stored (created automatically)
- `.vscode/`: VS Code configuration files
- `.github/workflows/`: GitHub Actions CI/CD configuration

## Running the Application

### Basic Usage

The main entry point for the pipeline is `src/main.py`:

```bash
python src/main.py
```

This will run the entire pipeline with default settings, using the data in the `data/` directory and storing results in the `output/` directory.

### Command-line Arguments

You can customize the execution with various command-line arguments:

- `--data-dir`: Directory containing the email data files (default: `./data/`)
- `--output-dir`: Directory to store all output files (default: `./output/`)
- `--skip-data-prep`: Skip data preparation step (use existing processed data)
- `--skip-analysis`: Skip summarization and classification step (use existing analysis results)
- `--skip-visualization`: Skip visualization step
- `--skip-stories`: Skip story development step

### Examples

Run only the data preparation step:
```bash
python src/main.py --skip-analysis --skip-visualization --skip-stories
```

Run visualization and story development using existing processed data and analysis results:
```bash
python src/main.py --skip-data-prep --skip-analysis
```

Use a different data directory:
```bash
python src/main.py --data-dir /path/to/enron/emails
```

You can also run individual modules directly:
```bash
python -m src.data_preparation.data_preparation
python -m src.summarization_classification.summarization_classification
python -m src.visualization.visualization
python -m src.story_development.story_development
```

## Development Workflow

### Code Style and Formatting

This project follows strict code style guidelines enforced by several tools:

1. **Black**: Code formatting
2. **isort**: Import sorting
3. **Ruff**: Linting
4. **mypy**: Type checking

You can run these tools manually:

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint code
ruff src tests

# Type check
mypy src
```

Or use the Makefile commands (see [Using the Makefile](#using-the-makefile) section).

### Writing Tests

All new code should be accompanied by tests. We use pytest for testing:

1. Create test files in the `tests/unit/` or `tests/integration/` directories
2. Name test files with the prefix `test_`
3. Name test functions with the prefix `test_`

Run the tests:
```bash
pytest
```

Or with coverage:
```bash
pytest --cov=src --cov-report=html
```

### Documentation

We use Sphinx for documentation:

1. Add docstrings to your code using Google style
2. Build the documentation:
   ```bash
   cd docs
   make html
   ```
3. View the documentation in `docs/build/html/index.html`

## Using the Makefile

The project includes a Makefile with several useful commands:

- `make help`: Display help information about available commands
- `make clean`: Remove build, test, coverage, and Python artifacts
- `make lint`: Check code style with ruff, black, and isort
- `make format`: Format code with black and isort
- `make test`: Run tests with pytest
- `make coverage`: Generate and view test coverage reports
- `make docs`: Generate Sphinx HTML documentation
- `make dist`: Build source and wheel package
- `make install`: Install the package
- `make dev-install`: Install the package in development mode

Example usage:
```bash
# Format code and run tests
make format
make test

# Generate documentation
make docs
```

## Debugging

### VS Code Debugging

The project includes VS Code launch configurations for debugging:

1. Open the project in VS Code
2. Go to the "Run and Debug" tab
3. Select one of the available configurations:
   - "Python: Current File": Debug the currently open file
   - "Python: Main Module": Debug the main.py entry point
   - "Python: Debug Tests": Debug test files

### Debugging from Command Line

You can also use Python's built-in debugger:

```bash
python -m pdb src/main.py
```

## Contributing Guidelines

### Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure they follow the code style guidelines

3. Run the tests to make sure everything works:
   ```bash
   make test
   ```

4. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

5. Push your branch to the remote repository:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a pull request on GitHub

### Code Review Process

All submissions will be reviewed by the project maintainers:

1. Automated checks must pass (linting, tests, etc.)
2. At least one maintainer must approve the changes
3. Changes should be rebased on the latest main branch before merging

For more details, see the [Contributing Guide](docs/source/contributing.rst).
