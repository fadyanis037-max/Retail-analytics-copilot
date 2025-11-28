import dspy
from dspy.teleprompt import BootstrapFewShot
from agent.dspy_signatures import SQLGeneratorSignature
from agent.tools.sqlite_tool import SQLiteTool
import json

# Setup DSPy
def setup_dspy():
    lm = dspy.LM(model='ollama/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://127.0.0.1:11434', api_key='')
    dspy.settings.configure(lm=lm)

# Metric
def validate_sql(example, pred, trace=None):
    sqlite_tool = SQLiteTool()
    sql = pred.sql_query.replace("```sql", "").replace("```", "").strip()
    
    # Basic check: is it empty?
    if not sql:
        return False
        
    # Execution check
    result = sqlite_tool.execute_sql(sql)
    if result["error"]:
        return False
        
    # Optional: Check if result matches expected (if we had ground truth answers)
    # For now, we just check if it executes.
    return True

def main():
    setup_dspy()
    
    # Training Data (Small set)
    train_examples = [
        dspy.Example(
            question="How many customers are in Germany?",
            schema="Table: customers\n  - CustomerID (INTEGER)\n  - Country (TEXT)",
            constraints="",
            sql_query="SELECT COUNT(*) FROM customers WHERE Country = 'Germany'"
        ).with_inputs("question", "schema", "constraints"),
        dspy.Example(
            question="List products with unit price > 50",
            schema="Table: products\n  - ProductName (TEXT)\n  - UnitPrice (REAL)",
            constraints="",
            sql_query="SELECT ProductName, UnitPrice FROM products WHERE UnitPrice > 50"
        ).with_inputs("question", "schema", "constraints"),
        dspy.Example(
            question="Total revenue for order 10248",
            schema="Table: order_items\n  - OrderID (INTEGER)\n  - UnitPrice (REAL)\n  - Quantity (INTEGER)\n  - Discount (REAL)",
            constraints="",
            sql_query="SELECT SUM(UnitPrice * Quantity * (1 - Discount)) FROM order_items WHERE OrderID = 10248"
        ).with_inputs("question", "schema", "constraints"),
        dspy.Example(
            question="What are the top 5 products by unit price?",
            schema="Table: products\n  - ProductName (TEXT)\n  - UnitPrice (REAL)",
            constraints="",
            sql_query="SELECT ProductName, UnitPrice FROM products ORDER BY UnitPrice DESC LIMIT 5"
        ).with_inputs("question", "schema", "constraints"),
    ]

    # Define the module to optimize
    class SQLGenModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate = dspy.ChainOfThought(SQLGeneratorSignature)
            
        def forward(self, question, schema, constraints):
            return self.generate(question=question, schema=schema, constraints=constraints)

    # Compile
    print("Optimizing SQL Generator...")
    teleprompter = BootstrapFewShot(metric=validate_sql, max_bootstrapped_demos=4, max_labeled_demos=4)
    optimized_program = teleprompter.compile(SQLGenModule(), trainset=train_examples)
    
    # Save
    optimized_program.save("agent/optimized_sql_gen.json")
    print("Optimization complete. Saved to agent/optimized_sql_gen.json")

if __name__ == "__main__":
    main()
