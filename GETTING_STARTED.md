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
- Node.js 18 or higher
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
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

For developers (includes development tools):

```bash
# Install Python package with development dependencies
pip install -e ".[dev]"

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Step 4: Install Pre-commit Hooks

Pre-commit hooks help ensure code quality before committing changes:

```bash
pre-commit install
```

## Understanding the Project Structure

The project is organized as follows:

### Backend (`src/`)

- `main.py`: Main entry point for the pipeline
- `data_preparation/`: Module for loading and preprocessing email data
- `summarization_classification/`: Module for topic modeling, clustering, and analysis
- `visualization/`: Module for creating visualizations
- `story_development/`: Module for generating narratives from the data
- `utils/`: Utility functions used across modules

### Frontend (`frontend/`)

- `app/`: Next.js app directory
  - `page.tsx`: Main page component
  - `layout.tsx`: Root layout component
  - `api/`: API route handlers
- `components/`: Reusable React components
  - `ui/`: Basic UI components
  - `features/`: Feature-specific components
- `lib/`: Utility functions and API clients
- `db/`: Database models and utilities
- `public/`: Static assets
- `styles/`: Global styles and CSS modules

### Other Directories

- `tests/`: Test files
  - `unit/`: Unit tests for individual components
  - `integration/`: Integration tests for module interactions
- `data/`: Directory containing the Enron email dataset
- `output/`: Directory where all results will be stored
- `.github/`: GitHub Actions workflows and templates
- `.vscode/`: VS Code configuration files

## Running the Application

### Backend

The main entry point for the pipeline is `src/main.py`:

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

#### Examples

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

#### Frontend Development

- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm run start`: Start production server
- `npm run lint`: Run ESLint
- `npm run test`: Run tests
- `npm run type-check`: Run TypeScript type checking

## Development Workflow

### Code Style and Formatting

This project follows strict code style guidelines enforced by several tools:

#### Backend

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
ruff check src tests

# Type check
mypy src
```

#### Frontend

1. **ESLint**: JavaScript/TypeScript linting
2. **Prettier**: Code formatting
3. **TypeScript**: Static type checking

```bash
cd frontend
npm run lint
npm run format
npm run type-check
```

### Writing Tests

#### Backend Tests

All new backend code should be accompanied by tests. We use pytest for testing:

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

#### Frontend Tests

Frontend tests use Jest and React Testing Library:

1. Create test files with `.test.tsx` or `.spec.tsx` extension
2. Place test files next to the components they test
3. Use React Testing Library for component testing

Run the tests:

```bash
cd frontend
npm test
```

### Documentation

#### Backend Documentation

We use Sphinx for backend documentation:

1. Add docstrings to your code using Google style
2. Build the documentation:
   ```bash
   cd docs
   make html
   ```
3. View the documentation in `docs/build/html/index.html`

#### Frontend Documentation

The frontend uses Next.js documentation:

1. Add JSDoc comments to your components and functions
2. Use TypeScript types and interfaces for better documentation
3. Add README.md files in component directories for complex components

## Using the Makefile

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

## Debugging

### VS Code Debugging

The project includes VS Code launch configurations for debugging both backend and frontend:

1. Open the project in VS Code
2. Go to the "Run and Debug" tab
3. Select the appropriate configuration:
   - "Python: Current File" for backend debugging
   - "Next.js: debug server-side" for frontend server debugging
   - "Next.js: debug client-side" for frontend client debugging

### Browser Developer Tools

For frontend debugging:

1. Open Chrome DevTools (F12 or Cmd+Option+I)
2. Use the React Developer Tools extension
3. Use the Network tab to debug API calls
4. Use the Console tab for JavaScript debugging

## Contributing Guidelines

1. Create a new branch for your feature or bugfix
2. Follow the code style guidelines
3. Write tests for new code
4. Update documentation
5. Submit a pull request

For more details, see the [Contributing Guide](docs/source/contributing.rst).
