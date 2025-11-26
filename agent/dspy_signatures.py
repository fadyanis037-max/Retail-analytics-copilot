"""
DSPy signatures and modules for the Retail Analytics Copilot.
Includes Router, NL2SQL, and Synthesizer modules.
"""

import dspy
from typing import Literal


# =============================================================================
# SIGNATURES
# =============================================================================

class RouterSignature(dspy.Signature):
    """Classify the type of question: rag, sql, or hybrid."""
    
    question = dspy.InputField(desc="The user's question")
    
    route = dspy.OutputField(
        desc="Classification: 'rag' (doc-only), 'sql' (db-only), or 'hybrid' (both)"
    )
    reasoning = dspy.OutputField(desc="Brief explanation for the classification")


class NL2SQLSignature(dspy.Signature):
    """Generate SQLite query from natural language and constraints."""
    
    question = dspy.InputField(desc="The user's question")
    schema = dspy.InputField(desc="Database schema information")
    constraints = dspy.InputField(
        desc="Extracted constraints: date ranges, entities, KPI formulas"
    )
    error_feedback = dspy.InputField(
        desc="Error from previous attempt (empty if first attempt)",
        default=""
    )
    
    sql = dspy.OutputField(desc="SQLite query to answer the question")
    explanation = dspy.OutputField(desc="Brief explanation of query logic")


class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from retrieval and SQL results."""
    
    question = dspy.InputField(desc="The user's question")
    format_hint = dspy.InputField(
        desc="Expected output format (e.g., 'int', 'float', '{category:str, quantity:int}')"
    )
    retrieval_results = dspy.InputField(
        desc="Document chunks with scores and IDs",
        default=""
    )
    sql_results = dspy.InputField(
        desc="SQL query results as JSON",
        default=""
    )
    
    final_answer = dspy.OutputField(
        desc="Answer matching format_hint exactly"
    )
    confidence = dspy.OutputField(
        desc="Confidence score 0.0-1.0"
    )
    explanation = dspy.OutputField(
        desc="Brief explanation (≤2 sentences)"
    )
    citations = dspy.OutputField(
        desc="Comma-separated list of DB tables and doc chunk IDs used"
    )


# =============================================================================
# MODULES
# =============================================================================

class RouterModule(dspy.Module):
    """Router module to classify questions."""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question: str):
        """
        Classify question type.
        
        Args:
            question: User's question
            
        Returns:
            dspy.Prediction with route and reasoning
        """
        result = self.predict(question=question)
        
        # Normalize route to one of: rag, sql, hybrid
        route = result.route.lower().strip()
        if route not in ['rag', 'sql', 'hybrid']:
            # Default to hybrid if unclear
            route = 'hybrid'
        
        return dspy.Prediction(
            route=route,
            reasoning=result.reasoning
        )


class NL2SQLModule(dspy.Module):
    """Natural language to SQL module."""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(NL2SQLSignature)
    
    def forward(self, question: str, schema: str, constraints: str, error_feedback: str = ""):
        """
        Generate SQL query.
        
        Args:
            question: User's question
            schema: Database schema
            constraints: Extracted constraints (dates, entities, etc.)
            error_feedback: Error from previous attempt (for repair)
            
        Returns:
            dspy.Prediction with sql and explanation
        """
        result = self.predict(
            question=question,
            schema=schema,
            constraints=constraints,
            error_feedback=error_feedback or "None"
        )
        
        # Clean up SQL (remove markdown code fences if present)
        sql = result.sql.strip()
        if sql.startswith('```'):
            # Remove code fences
            lines = sql.split('\n')
            sql = '\n'.join(
                line for line in lines 
                if not line.strip().startswith('```')
            )
            sql = sql.strip()
        
        return dspy.Prediction(
            sql=sql,
            explanation=result.explanation
        )


class SynthesizerModule(dspy.Module):
    """Synthesizer module to format final answers."""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(SynthesizerSignature)
    
    def forward(
        self,
        question: str,
        format_hint: str,
        retrieval_results: str = "",
        sql_results: str = ""
    ):
        """
        Synthesize final answer from available information.
        
        Args:
            question: User's question
            format_hint: Expected output format
            retrieval_results: Document retrieval results
            sql_results: SQL query results
            
        Returns:
            dspy.Prediction with final_answer, confidence, explanation, citations
        """
        result = self.predict(
            question=question,
            format_hint=format_hint,
            retrieval_results=retrieval_results or "No retrieval results",
            sql_results=sql_results or "No SQL results"
        )
        
        return dspy.Prediction(
            final_answer=result.final_answer,
            confidence=result.confidence,
            explanation=result.explanation,
            citations=result.citations
        )
