Contributing
============

We welcome contributions to the Enron Email Analysis Pipeline! This document provides guidelines and instructions for contributing.

Development Setup
----------------

1. Fork the repository
2. Clone your fork:
   .. code-block:: bash

      git clone https://github.com/yourusername/enron-email-analysis.git
      cd enron-email-analysis

3. Set up the development environment:
   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      pip install -r requirements.txt
      pip install -r requirements-dev.txt

4. Set up pre-commit hooks:
   .. code-block:: bash

      pre-commit install

Code Style
----------

We use several tools to maintain code quality:

Backend (Python):
- Black for code formatting
- isort for import sorting
- Ruff for linting
- mypy for type checking

Frontend (TypeScript/React):
- ESLint for linting
- Prettier for code formatting
- TypeScript for type checking

Running Tests
------------

Backend tests:
.. code-block:: bash

   pytest

Frontend tests:
.. code-block:: bash

   cd frontend
   npm test

Documentation
------------

We use Sphinx for documentation. To build the docs:

.. code-block:: bash

   cd docs
   make html

Pull Request Process
------------------

1. Create a new branch for your feature/fix
2. Make your changes
3. Run tests and ensure they pass
4. Update documentation if needed
5. Submit a pull request

For more details, see our :doc:`getting_started` guide.
