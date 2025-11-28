import dspy
from typing import Literal

class RouterSignature(dspy.Signature):
    """Classify the user question to determine the best strategy: 'sql', 'rag', or 'hybrid'."""
    
    question = dspy.InputField(desc="The user's question about retail analytics.")
    strategy = dspy.OutputField(desc="The best strategy to answer the question. Options: 'sql', 'rag', 'hybrid'.")

class SQLGeneratorSignature(dspy.Signature):
    """Generate a SQLite query based on the question and schema."""
    
    question = dspy.InputField(desc="The user's question.")
    schema = dspy.InputField(desc="The database schema definitions.")
    constraints = dspy.InputField(desc="Any specific constraints (dates, categories, etc.) extracted from documents.")
    sql_query = dspy.OutputField(desc="The valid SQLite query to answer the question.")

class SynthesizerSignature(dspy.Signature):
    """Synthesize a final answer based on the question, SQL results, and retrieved context."""
    
    question = dspy.InputField(desc="The user's question.")
    context = dspy.InputField(desc="Retrieved document chunks.")
    sql_result = dspy.InputField(desc="The result of the executed SQL query.")
    format_hint = dspy.InputField(desc="The expected format of the answer (e.g., int, float, list[dict]).")
    
    final_answer = dspy.OutputField(desc="The final answer matching the format hint.")
    explanation = dspy.OutputField(desc="A brief explanation (<= 2 sentences).")
    citations = dspy.OutputField(desc="List of DB tables and doc chunks used.")

class PlannerSignature(dspy.Signature):
    """Extract constraints and entities from the question and context."""
    
    question = dspy.InputField(desc="The user's question.")
    context = dspy.InputField(desc="Retrieved document chunks.")
    constraints = dspy.OutputField(desc="Extracted constraints (e.g., date ranges, specific products/categories).")
