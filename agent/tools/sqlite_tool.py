import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path

    def get_schema(self, tables: Optional[List[str]] = None) -> str:
        """
        Returns the schema for the specified tables (or all if None).
        Includes table names and column definitions.
        """
        if tables is None:
            tables = ["orders", "order_items", "products", "customers", "categories", "suppliers"]
        
        schema_str = ""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for table in tables:
                # Check if table/view exists
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name='{table}'")
                if not cursor.fetchone():
                    continue
                    
                schema_str += f"Table: {table}\n"
                cursor.execute(f"PRAGMA table_info('{table}')")
                columns = cursor.fetchall()
                for col in columns:
                    # cid, name, type, notnull, dflt_value, pk
                    schema_str += f"  - {col[1]} ({col[2]})\n"
                schema_str += "\n"
                
            conn.close()
        except Exception as e:
            return f"Error getting schema: {e}"
            
        return schema_str

    def execute_sql(self, query: str) -> Dict[str, Any]:
        """
        Executes a SQL query and returns the results.
        Returns a dict with 'columns', 'rows', and 'error'.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            # Use pandas for easy execution and fetching
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return {
                "columns": list(df.columns),
                "rows": df.to_dict(orient="records"),
                "error": None
            }
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "error": str(e)
            }
