import os
from typing import Optional

import pandas as pd
import psycopg2
from loguru import logger
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, text


class DatabaseManager:
    """Database manager class to handle database initialization and operations."""

    def __init__(self):
        """
        Initialize the database manager using environment variables.
        """
        # Try to get DATABASE_URL first
        self.db_url = os.getenv("DATABASE_URL")
        logger.debug(f"Found DATABASE_URL: {self.db_url}")

        # If DATABASE_URL is not set, construct it from individual environment variables
        if not self.db_url:
            pg_host = os.getenv("PG_HOST") or os.getenv("PGHOST")
            pg_port = os.getenv("PG_PORT", "5432")
            pg_user = os.getenv("PG_USER") or os.getenv("PGUSER")
            pg_password = os.getenv("PG_PASSWORD") or os.getenv("PGPASSWORD")
            pg_database = os.getenv("PG_DATABASE") or os.getenv("PGDATABASE")

            logger.debug(f"Environment variables found:")
            logger.debug(f"PG_HOST/PGHOST: {pg_host}")
            logger.debug(f"PG_PORT: {pg_port}")
            logger.debug(f"PG_USER/PGUSER: {pg_user}")
            logger.debug(
                f"PG_PASSWORD/PGPASSWORD: {'*' * len(pg_password) if pg_password else None}"
            )
            logger.debug(f"PG_DATABASE/PGDATABASE: {pg_database}")

            if all([pg_host, pg_user, pg_password, pg_database]):
                self.db_url = (
                    f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
                )
                logger.debug(f"Constructed database URL: {self.db_url}")
            else:
                missing_vars = []
                if not pg_host:
                    missing_vars.append("host")
                if not pg_user:
                    missing_vars.append("user")
                if not pg_password:
                    missing_vars.append("password")
                if not pg_database:
                    missing_vars.append("database")
                raise ValueError(
                    f"Missing required database connection parameters: {', '.join(missing_vars)}"
                )

        # Parse the database URL to get individual components
        self.conn = None
        self.cursor = None
        self.engine = None

    def connect(self, db_name: Optional[str] = None) -> bool:
        """
        Connect to the database.

        Args:
            db_name (str, optional): Database name to connect to. If None, uses the database from DATABASE_URL.

        Returns:
            bool: True if the connection is successful, False otherwise
        """
        try:
            # If db_name is provided, modify the connection URL
            if db_name:
                # Parse the current URL and replace the database name
                from urllib.parse import urlparse

                parsed = urlparse(self.db_url)
                path_parts = parsed.path.split("/")
                path_parts[-1] = db_name
                modified_url = parsed._replace(path="/".join(path_parts)).geturl()
                self.conn = psycopg2.connect(modified_url)
            else:
                self.conn = psycopg2.connect(self.db_url)

            self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def drop_all_tables(self) -> bool:
        """
        Drop all tables in the database.

        Returns:
            bool: True if tables dropped successfully, False otherwise
        """
        try:
            if not self.connect():
                return False

            # Get all table names
            self.cursor.execute(
                """
                                SELECT table_name
                                FROM information_schema.tables
                                WHERE table_schema = 'public'
                                """
            )
            tables = self.cursor.fetchall()

            # Drop each table
            for table in tables:
                stmt = f"DROP TABLE IF EXISTS {table[0]} CASCADE;"
                self.cursor.execute(stmt)

            logger.info("Dropped all existing tables")
            self.close()
            return True
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            return False

    def initialize_schema(self) -> bool:
        """
        Initialize the database schema by executing the SQL schema file.

        Returns:
            bool: True if schema initialized successfully, False otherwise
        """
        try:
            if not self.connect():
                return False

            # Read and execute the schema file
            schema_path = os.path.join(os.path.dirname(__file__), "database_schema.sql")
            with open(schema_path, "r") as f:
                schema_sql = f.read()
                self.cursor.execute(schema_sql)

            logger.info("Database schema initialized successfully")
            self.close()
            return True
        except Exception as e:
            logger.error(f"Error initializing schema: {e}")
            return False

    def reset_db(self) -> bool:
        """
        Initialize the database by setting up the schema.
        This will drop all existing tables and recreate them.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if not self.connect():
                return False

            # Drop all existing tables
            if not self.drop_all_tables():
                return False

            # Initialize schema
            if not self.initialize_schema():
                return False

            logger.success("Database reset successfull")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False

    def save_df_to_db(
        self, table_name: str, df: pd.DataFrame, suc_msg: str = None, if_exists="fail"
    ) -> bool:
        """
        Save a pandas DataFrame to a PostgreSQL table.
        Args:
            table_name (str): Name of the table to save.
            df (pd.DataFrame): Pandas DataFrame to save.
            suc_msg (str): Log success message after saving the DataFrame to the database successfully.
            if_exists (str, optional): What to do if the table already exists. options are ('fail', 'replace', or 'append'). Defaults to "fail".
        Returns:
            bool: True if table was successfully saved, False otherwise
        """
        try:
            if self.engine is None:
                self.engine = create_engine(self.db_url)

            df_copy = df.copy()

            # Convert complex data types to strings for PostgreSQL compatibility
            for col in df_copy.columns:
                if df_copy[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df_copy[col] = df_copy[col].apply(
                        lambda x: str(x) if isinstance(x, (list, dict)) else x
                    )

            # If replacing table, first drop it with CASCADE
            if if_exists == "replace":
                with self.engine.connect() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                    conn.commit()

            df_copy.to_sql(
                table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False,
                method="multi",
                chunksize=1000,
            )

            if suc_msg is None:
                suc_msg = "Successfully saved DataFrame to database"

            logger.success(suc_msg)
            return True
        except Exception as e:
            logger.error(f"Error saving DataFrame to database: {e}")
            return False
