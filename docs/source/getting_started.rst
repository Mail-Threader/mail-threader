Getting Started
==============

This guide will help you get started with the Enron Email Analysis Pipeline.

Installation
-----------

1. Clone the repository:
   .. code-block:: bash

      git clone https://github.com/yourusername/enron-email-analysis.git
      cd enron-email-analysis

2. Create and activate a virtual environment:
   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   .. code-block:: bash

      pip install -r requirements.txt

4. Install frontend dependencies:
   .. code-block:: bash

      cd frontend
      npm install

Running the Application
---------------------

1. Start the backend pipeline:
   .. code-block:: bash

      python src/main.py

2. Start the frontend development server:
   .. code-block:: bash

      cd frontend
      npm run dev

The application will be available at:
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000

Command Line Arguments
--------------------

The pipeline supports several command line arguments:

- ``--skip``: Skip specific steps in the pipeline
  Example: ``python src/main.py --skip data_preparation visualization``

- ``--run``: Run only specific steps
  Example: ``python src/main.py --run sentiment_analysis``

For more information about available steps and options, see the :doc:`modules/index` documentation.
