# Enron Email Analysis Pipeline

This project provides a comprehensive pipeline for analyzing the Enron email dataset. It includes modules for data preparation, summarization/classification, visualization, and story development.

## Project Structure

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

## Prerequisites

- Python 3.9 or higher
- Required Python packages (listed in requirements.txt)

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
pip install -r requirements.txt
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

4. Install pre-commit hooks:

```bash
pre-commit install
```

## How to Run

The main entry point for the pipeline is `src/main.py`. You can run it with various command-line arguments to customize the execution.

### Basic Usage

```bash
python src/main.py
```

This will run the entire pipeline with default settings, using the data in the `data/` directory and storing results in the `output/` directory.

### Command-line Arguments

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

## Output

The pipeline creates the following output directories:

- `output/processed_data/`: Contains processed email data
- `output/analysis_results/`: Contains results from summarization and classification
- `output/visualizations/`: Contains all generated visualizations
- `output/stories/`: Contains generated stories and narratives

A final HTML report is also generated in the main output directory, providing links to all results.

## Individual Modules

You can also run each module individually:

```bash
python -m src.data_preparation.data_preparation
python -m src.summarization_classification.summarization_classification
python -m src.visualization.visualization
python -m src.story_development.story_development
```

Each module will look for the necessary input files from previous steps and generate its own outputs.

## Development

### Code Style

This project uses the following tools to ensure code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Linting
- **mypy**: Type checking

You can run these tools using VS Code tasks or from the command line:

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

### Testing

We use pytest for testing. To run the tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=src --cov-report=html
```

### Documentation

We use Sphinx for documentation. To build the documentation:

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`.

### VS Code Integration

This project includes VS Code configuration files:

- `settings.json`: Editor settings, Python settings, linting, formatting
- `launch.json`: Debug configurations
- `tasks.json`: Common tasks like testing, linting, formatting

### Continuous Integration

GitHub Actions workflows are set up to run on each push and pull request:

- Linting and type checking
- Testing on multiple Python versions
- Building the package
- Building documentation

## Contributing

Please see the [Contributing Guide](docs/source/contributing.rst) for details on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
