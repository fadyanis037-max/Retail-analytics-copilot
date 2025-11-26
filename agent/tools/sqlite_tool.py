"""SQLite database tools for Northwind database access."""

import sqlite3
from typing import List, Dict, Any, Tuple, Optional
from contextlib import contextmanager


class SQLiteTool:
    """Utility class for SQLite database operations."""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite tool.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def get_schema(self) -> str:
        """
        Get database schema as a formatted string.
        Includes table names and column information.
        
        Returns:
            Formatted schema string for use in prompts
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_parts = ["# Northwind Database Schema\n"]
            
            for table in tables:
                # Get column info for each table - quote table name properly
                quoted_table = f'"{table}"' if ' ' in table else table
                cursor.execute(f"PRAGMA table_info({quoted_table})")
                columns = cursor.fetchall()
                
                schema_parts.append(f"\n## Table: {table}")
                schema_parts.append("Columns:")
                for col in columns:
                    col_name = col[1]
                    col_type = col[2]
                    is_pk = " (PRIMARY KEY)" if col[5] else ""
                    schema_parts.append(f"  - {col_name}: {col_type}{is_pk}")
            
            return "\n".join(schema_parts)
    
    def execute_query(self, sql: str) -> Tuple[bool, Any, Optional[str]]:
        """
        Execute a SQL query and return results or error.
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (success, results, error_message)
            - success: True if query executed successfully
            - results: List of dicts for SELECT, affected rows for others
            - error_message: Error description if success is False
        """
        # Basic safety check - only allow SELECT statements
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith('SELECT') and not sql_stripped.startswith('WITH'):
            return False, None, "Only SELECT queries are allowed for safety"
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                # Fetch all results and convert to list of dicts
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                
                return True, results, None
                
        except sqlite3.Error as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    def validate_query(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL query without executing it.
        
        Args:
            sql: SQL query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        sql_upper = sql.upper()
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False, f"Query contains forbidden keyword: {keyword}"
        
        # Try to execute with EXPLAIN to check syntax
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"EXPLAIN {sql}")
                return True, None
        except sqlite3.Error as e:
            return False, str(e)
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict]:
        """
        Get sample rows from a table.
        
        Args:
            table_name: Name of the table
            limit: Number of rows to fetch
            
        Returns:
            List of row dictionaries
        """
        success, results, error = self.execute_query(
            f"SELECT * FROM {table_name} LIMIT {limit}"
        )
        return results if success else []
