import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from loguru import logger
from utils import save_error_log
from datetime import datetime


class DatabaseManager:
	"""Database manager class to handle database initialization and operations."""

	def __init__(self, db_name="mail_threader"):
		"""
        Initialize the database manager using environment variables.
        """
		self.db_name = db_name
		self.engine = None
		self.Session = None
		self.connection_url = self._get_connection_url()
		self.connect()

	def _get_connection_url(self):
		"""
        Get the database connection URL from environment variables.
        """
		db_user = os.getenv("DB_USER", "postgres")
		db_password = os.getenv("DB_PASSWORD", "password")
		db_host = os.getenv("DB_HOST", "localhost")
		db_port = os.getenv("DB_PORT", "5432")
		return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{self.db_name}"

	def connect(self):
		"""
        Establish the database connection and create the database if it doesn't exist.
        """
		try:
			# Connect to the default postgres database to create the new database
			default_engine = create_engine(
				self.connection_url.replace(f"/{self.db_name}", "/postgres"))
			with default_engine.connect() as connection:
				connection.execution_options(isolation_level="AUTOCOMMIT")
				# check if the database already exists
				result = connection.execute(
					text(f"""
					SELECT 1 FROM pg_database WHERE datname = '{self.db_name}'
				"""))
				if not result.fetchone():
					# If the database does not exist, create it
					logger.info(f"Creating database '{self.db_name}'...")
					connection.execute(text(f"CREATE DATABASE {self.db_name}"))
				else:
					logger.info(f"Database '{self.db_name}' already exists. Skipping creation.")

				logger.info(f"Database '{self.db_name}' created successfully.")
		except OperationalError:
			logger.info(f"Database '{self.db_name}' already exists.")
		except Exception as e:
			logger.error(f"Failed to create database: {e}")
			save_error_log(f"Failed to create database: {e}")
			return

		try:
			self.engine = create_engine(self.connection_url)
			self.Session = sessionmaker(bind=self.engine)
			logger.info("Database connection established successfully.")
		except Exception as e:
			logger.error(f"Failed to connect to database: {e}")
			save_error_log(f"Failed to connect to database: {e}")

	def get_session(self):
		"""
        Get a new database session.

        Returns:
            sqlalchemy.orm.Session: A new database session.
        """
		if self.Session is None:
			self.connect()

		if self.Session is None:
			logger.error("Database engine is not initialized. Cannot create session.")
			save_error_log("Database engine is not initialized. Cannot create session.")
			return None

		return self.Session()

	def create_tables(self):
		"""
        Create tables from the database_schema_config.sql file.
        Checks if existing schema matches the file schema and makes changes if needed.
        """
		if not self.engine:
			self.connect()

		if self.engine is None:
			logger.error("Database engine is not initialized. Cannot create tables.")
			save_error_log("Database engine is not initialized. Cannot create tables.")
			return

		try:
			with self.engine.connect() as connection:
				schema_path = os.path.join(os.path.dirname(__file__), "database_schema_config.sql")
				with open(schema_path, "r") as f:
					schema = f.read()

					# execute the schema file to create tables
					logger.info("Creating/updating database tables...")
					connection.execute(text(schema))
				connection.commit()
				logger.info("Database tables created/updated successfully.")

				# # Check if schema changes are needed
				# if self._schema_needs_update(connection, schema):
				# 	logger.info("Schema differences detected. Updating database schema...")
				# 	self._update_schema(connection, schema)
				# else:
				# 	logger.info("Database schema is up to date.")

		except Exception as e:
			logger.error(f"Failed to create/update tables: {e}")
			save_error_log(f"Failed to create/update tables: {e}")

	def _schema_needs_update(self, connection, schema_content):
		"""
        Check if the current database schema needs to be updated.

        Args:
            connection: Database connection object
            schema_content: Content of the schema file

        Returns:
            bool: True if schema needs update, False otherwise
        """
		try:
			# Get current table information
			current_tables = self._get_current_table_info(connection)
			expected_tables = self._parse_schema_tables(schema_content)

			# Compare table structures
			return not self._schemas_match(current_tables, expected_tables)
		except Exception as e:
			logger.warning(f"Could not check schema differences: {e}")
			return True  # Default to updating if we can't check

	def _get_current_table_info(self, connection):
		"""
        Get information about current tables in the database.

        Args:
            connection: Database connection object

        Returns:
            dict: Dictionary containing table information
        """
		tables_info = {}

		# Get all tables
		result = connection.execute(
			text("""
			SELECT table_name
			FROM information_schema.tables
			WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
		"""))

		for row in result:
			table_name = row[0]

			# Get columns for each table
			column_result = connection.execute(
				text(f"""
				SELECT column_name, data_type, is_nullable, column_default
				FROM information_schema.columns
				WHERE table_name = '{table_name}' AND table_schema = 'public'
				ORDER BY ordinal_position
			"""))

			tables_info[table_name] = {
				'columns': [(col[0], col[1], col[2], col[3]) for col in column_result]
			}

		return tables_info

	def _parse_schema_tables(self, schema_content):
		"""
        Parse the schema file to extract expected table structure.

        Args:
            schema_content: Content of the schema file

        Returns:
            dict: Dictionary containing expected table information
        """
		# This is a simplified parser - in a production environment,
		# you might want to use a proper SQL parser
		expected_tables = {}

		# Extract table names from CREATE TABLE statements
		import re
		table_pattern = r'CREATE TABLE IF NOT EXISTS (\w+)'
		tables = re.findall(table_pattern, schema_content, re.IGNORECASE)

		for table in tables:
			expected_tables[table] = {'exists': True}

		return expected_tables

	def _schemas_match(self, current_tables, expected_tables):
		"""
        Compare current and expected table structures.

        Args:
            current_tables: Current database table information
            expected_tables: Expected table information from schema file

        Returns:
            bool: True if schemas match, False otherwise
        """
		# Check if all expected tables exist
		for expected_table in expected_tables:
			if expected_table not in current_tables:
				logger.info(f"Table '{expected_table}' is missing from database")
				return False

		# In a more comprehensive implementation, you would also check:
		# - Column types and constraints
		# - Indexes
		# - Foreign keys
		# - Triggers and functions

		return True

	def _update_schema(self, connection, schema_content):
		"""
        Update the database schema by executing the schema file.

        Args:
            connection: Database connection object
            schema_content: Content of the schema file
        """
		try:
			# Split schema into individual statements
			statements = [stmt.strip() for stmt in schema_content.split(';') if stmt.strip()]

			for statement in statements:
				if statement:
					connection.execute(text(statement))

			connection.commit()
			logger.info("Database schema updated successfully.")
		except Exception as e:
			connection.rollback()
			logger.error(f"Failed to update schema: {e}")
			save_error_log(f"Failed to update schema: {e}")
			raise

	def insert_from_dataframe(self, df: pd.DataFrame, table_name: str, columns: list[str]):
		"""
		Insert rows from a DataFrame into a specified table.

		Args:
			df (pd.DataFrame): The DataFrame to insert.
			table_name (str): The name of the table to insert into.
			columns (list[str]): The list of columns to insert.
		"""
		if not self.engine:
			self.connect()

		if self.engine is None:
			logger.error("Database engine is not initialized. Cannot insert data.")
			save_error_log("Database engine is not initialized. Cannot insert data.")
			return

		if df.empty:
			logger.warning("DataFrame is empty. No data to insert.")
			return

		# Add result_type to the list of columns to insert
		columns_to_insert = columns + ["result_type"]

		# Ensure all required columns are in the DataFrame
		if not all(col in df.columns for col in columns):
			missing_cols = [col for col in columns if col not in df.columns]
			logger.error(f"Missing columns in DataFrame: {missing_cols}")
			raise ValueError(f"Missing columns: {missing_cols}")

		result_type = f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
		df['result_type'] = result_type

		try:
			# Select only the columns that exist in the table
			df_to_insert = df[columns_to_insert]
			df_to_insert.to_sql(table_name, self.engine, if_exists='append', index=False)
			logger.info(f"Successfully inserted {len(df_to_insert)} rows into '{table_name}'.")
		except Exception as e:
			logger.error(f"Failed to insert data into '{table_name}': {e}")
			save_error_log(f"Failed to insert data into '{table_name}': {e}")

	def validate_schema(self):
		"""
        Validate the current database schema against the schema file.

        Returns:
            dict: Validation results including missing tables, columns, and other issues
        """
		if not self.engine:
			self.connect()

		if self.engine is None:
			logger.error("Database engine is not initialized. Cannot validate schema.")
			return {"error": "Database engine not initialized"}

		try:
			with self.engine.connect() as connection:
				schema_path = os.path.join(os.path.dirname(__file__), "database_schema_config.sql")
				with open(schema_path, "r") as f:
					expected_schema = f.read()

				current_tables = self._get_current_table_info(connection)
				expected_tables = self._parse_schema_tables(expected_schema)

				validation_results = {
					"missing_tables": [],
					"extra_tables": [],
					"schema_valid": True,
					"recommendations": []
				}

				# Check for missing tables
				for table_name in expected_tables:
					if table_name not in current_tables:
						validation_results["missing_tables"].append(table_name)
						validation_results["schema_valid"] = False

				# Check for extra tables (not in schema file)
				for table_name in current_tables:
					if table_name not in expected_tables:
						validation_results["extra_tables"].append(table_name)

				# Add recommendations
				if validation_results["missing_tables"]:
					validation_results["recommendations"].append(
						"Run create_tables() to create missing tables")

				if validation_results["extra_tables"]:
					validation_results["recommendations"].append(
						"Consider removing unused tables: " +
						", ".join(validation_results["extra_tables"]))

				return validation_results

		except Exception as e:
			logger.error(f"Failed to validate schema: {e}")
			return {"error": f"Schema validation failed: {e}"}

	def get_schema_info(self):
		"""
        Get detailed information about the current database schema.

        Returns:
            dict: Detailed schema information
        """
		if not self.engine:
			self.connect()

		if self.engine is None:
			logger.error("Database engine is not initialized. Cannot get schema info.")
			return {"error": "Database engine not initialized"}

		try:
			with self.engine.connect() as connection:
				schema_info = {"tables": {}, "indexes": {}, "functions": []}

				# Get table information
				current_tables = self._get_current_table_info(connection)

				for table_name, table_info in current_tables.items():
					# Get row count
					count_result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
					count_row = count_result.fetchone()
					row_count = count_row[0] if count_row else 0

					# Get indexes for this table
					index_result = connection.execute(
						text(f"""
						SELECT indexname, indexdef
						FROM pg_indexes
						WHERE tablename = '{table_name}'
					"""))

					table_indexes = {row[0]: row[1] for row in index_result}

					schema_info["tables"][table_name] = {
						"columns": table_info["columns"],
						"row_count": row_count,
						"indexes": table_indexes
					}

				# Get function information
				function_result = connection.execute(
					text("""
					SELECT proname, prosrc
					FROM pg_proc
					WHERE pronamespace = (SELECT oid FROM pg_namespace WHERE nspname = 'public')
				"""))

				schema_info["functions"] = [{
					"name": row[0],
					"definition": row[1]
				} for row in function_result]

				return schema_info

		except Exception as e:
			logger.error(f"Failed to get schema info: {e}")
			return {"error": f"Schema info retrieval failed: {e}"}
