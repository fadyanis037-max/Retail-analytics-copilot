"""
LangGraph hybrid agent for Retail Analytics.
Orchestrates RAG + SQL with repair loop using DSPy modules.
"""

import os
import json
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import dspy

from agent.dspy_signatures import RouterModule, NL2SQLModule, SynthesizerModule
from agent.rag.retrieval import DocumentRetriever
from agent.tools.sqlite_tool import SQLiteTool


# =============================================================================
# STATE SCHEMA
# =============================================================================

class AgentState(TypedDict):
    """State for the hybrid agent graph."""
    question: str
    format_hint: str
    
    # Routing
    route: str  # rag, sql, or hybrid
    
    # Retrieval
    retrieval_results: List[Dict]  # chunks with IDs, scores
    retrieval_score: float
    
    # Planning
    constraints: str  # Extracted constraints (dates, entities, formulas)
    
    # SQL
    sql: str
    sql_results: List[Dict]
    sql_error: str
    sql_success: bool
    
    # Repair
    repair_count: int
    
    # Final output
    final_answer: Any
    confidence: float
    explanation: str
    citations: List[str]


# =============================================================================
# AGENT
# =============================================================================

class HybridAgent:
    """Hybrid RAG + SQL agent using LangGraph."""
    
    def __init__(
        self,
        docs_dir: str,
        db_path: str,
        model_name: str = "phi3.5:3.8b-mini-instruct-q4_K_M"
    ):
        """
        Initialize agent.
        
        Args:
            docs_dir: Path to docs directory
            db_path: Path to SQLite database
            model_name: Ollama model name
        """
        self.docs_dir = docs_dir
        self.db_path = db_path
        
        # Initialize DSPy with Ollama
        self.lm = dspy.OllamaLocal(model=model_name, max_tokens=1000)
        dspy.settings.configure(lm=self.lm)
        
        # Initialize tools
        self.retriever = DocumentRetriever(docs_dir)
        self.retriever.load_and_index()
        
        self.db_tool = SQLiteTool(db_path)
        self.schema = self.db_tool.get_schema()
        
        # Initialize DSPy modules
        self.router_module = RouterModule()
        self.nl2sql_module = NL2SQLModule()
        self.synthesizer_module = SynthesizerModule()
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("nl2sql", self.nl2sql_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("repair", self.repair_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Router edges
        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "rag_only": "retriever",
                "sql_only": "planner",
                "hybrid": "retriever"  # hybrid does both retriever and planner
            }
        )
        
        # Retriever edges
        workflow.add_conditional_edges(
            "retriever",
            self.after_retriever,
            {
                "to_planner": "planner",  # for hybrid
                "to_synthesizer": "synthesizer"  # for rag-only
            }
        )
        
        # Planner -> NL2SQL
        workflow.add_edge("planner", "nl2sql")
        
        # NL2SQL -> Executor
        workflow.add_edge("nl2sql", "executor")
        
        # Executor edges (success or repair)
        workflow.add_conditional_edges(
            "executor",
            self.after_executor,
            {
                "to_synthesizer": "synthesizer",
                "to_repair": "repair"
            }
        )
        
        # Repair edges (retry or give up)
        workflow.add_conditional_edges(
            "repair",
            self.after_repair,
            {
                "retry": "nl2sql",
                "give_up": "synthesizer"
            }
        )
        
        # Synthesizer -> END
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    # =========================================================================
    # NODES
    # =========================================================================
    
    def router_node(self, state: AgentState) -> AgentState:
        """Route the question to rag, sql, or hybrid."""
        result = self.router_module(question=state["question"])
        state["route"] = result.route
        state["repair_count"] = 0
        return state
    
    def retriever_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant document chunks."""
        chunks = self.retriever.search(state["question"], top_k=3)
        
        state["retrieval_results"] = [
            {
                "id": chunk.id,
                "content": chunk.content,
                "source": chunk.source,
                "score": chunk.score
            }
            for chunk in chunks
        ]
        
        # Calculate avg retrieval score
        if chunks:
            state["retrieval_score"] = sum(c.score for c in chunks) / len(chunks)
        else:
            state["retrieval_score"] = 0.0
        
        return state
    
    def planner_node(self, state: AgentState) -> AgentState:
        """Extract constraints from retrieval results and question."""
        constraints_parts = []
        
        # Extract date ranges from marketing calendar
        for result in state.get("retrieval_results", []):
            if "marketing_calendar" in result["id"]:
                constraints_parts.append(f"Marketing calendar context: {result['content']}")
        
        # Extract KPI formulas
        for result in state.get("retrieval_results", []):
            if "kpi_definitions" in result["id"]:
                constraints_parts.append(f"KPI definition: {result['content']}")
        
        # Add question context
        constraints_parts.append(f"Question context: {state['question']}")
        
        state["constraints"] = "\n".join(constraints_parts) if constraints_parts else "No specific constraints"
        
        return state
    
    def nl2sql_node(self, state: AgentState) -> AgentState:
        """Generate SQL query using DSPy."""
        error_feedback = state.get("sql_error", "")
        
        result = self.nl2sql_module(
            question=state["question"],
            schema=self.schema,
            constraints=state.get("constraints", ""),
            error_feedback=error_feedback
        )
        
        state["sql"] = result.sql
        return state
    
    def executor_node(self, state: AgentState) -> AgentState:
        """Execute SQL query."""
        success, results, error = self.db_tool.execute_query(state["sql"])
        
        state["sql_success"] = success
        if success:
            state["sql_results"] = results
            state["sql_error"] = ""
        else:
            state["sql_results"] = []
            state["sql_error"] = error
        
        return state
    
    def repair_node(self, state: AgentState) -> AgentState:
        """Increment repair counter."""
        state["repair_count"] = state.get("repair_count", 0) + 1
        return state
    
    def synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final answer with citations."""
        # Format retrieval results
        retrieval_text = ""
        if state.get("retrieval_results"):
            retrieval_text = json.dumps([
                {"id": r["id"], "content": r["content"], "score": r["score"]}
                for r in state["retrieval_results"]
            ], indent=2)
        
        # Format SQL results
        sql_text = ""
        if state.get("sql_results"):
            sql_text = json.dumps(state["sql_results"], indent=2)
        
        result = self.synthesizer_module(
            question=state["question"],
            format_hint=state["format_hint"],
            retrieval_results=retrieval_text,
            sql_results=sql_text
        )
        
        # Parse final answer based on format_hint
        final_answer = self._parse_answer(result.final_answer, state["format_hint"])
        
        # Parse confidence
        try:
            confidence = float(result.confidence)
        except:
            confidence = 0.5
        
        # Calculate confidence heuristic
        confidence = self._calculate_confidence(state, confidence)
        
        # Parse citations
        citations = [c.strip() for c in result.citations.split(",") if c.strip()]
        
        state["final_answer"] = final_answer
        state["confidence"] = confidence
        state["explanation"] = result.explanation
        state["citations"] = citations
        
        return state
    
    # =========================================================================
    # ROUTING LOGIC
    # =========================================================================
    
    def route_decision(self, state: AgentState) -> str:
        """Determine routing after router node."""
        route = state["route"]
        if route == "rag":
            return "rag_only"
        elif route == "sql":
            return "sql_only"
        else:
            return "hybrid"
    
    def after_retriever(self, state: AgentState) -> str:
        """Determine routing after retriever."""
        if state["route"] == "hybrid":
            return "to_planner"
        else:
            return "to_synthesizer"
    
    def after_executor(self, state: AgentState) -> str:
        """Determine routing after executor."""
        if state["sql_success"]:
            return "to_synthesizer"
        else:
            # Try repair if under limit
            if state.get("repair_count", 0) < 2:
                return "to_repair"
            else:
                return "to_synthesizer"
    
    def after_repair(self, state: AgentState) -> str:
        """Determine routing after repair."""
        if state["repair_count"] < 2:
            return "retry"
        else:
            return "give_up"
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _parse_answer(self, answer_str: str, format_hint: str) -> Any:
        """Parse answer string according to format_hint."""
        answer_str = answer_str.strip()
        
        # Try to extract JSON if present
        if "{" in answer_str or "[" in answer_str:
            try:
                # Find JSON-like content
                start = min(
                    answer_str.find("{") if "{" in answer_str else len(answer_str),
                    answer_str.find("[") if "[" in answer_str else len(answer_str)
                )
                end = max(
                    answer_str.rfind("}") if "}" in answer_str else -1,
                    answer_str.rfind("]") if "]" in answer_str else -1
                ) + 1
                
                if start < end:
                    json_str = answer_str[start:end]
                    return json.loads(json_str)
            except:
                pass
        
        # Try int
        if format_hint == "int":
            try:
                # Extract first number
                import re
                match = re.search(r'-?\d+', answer_str)
                if match:
                    return int(match.group())
            except:
                pass
        
        # Try float
        if format_hint == "float":
            try:
                import re
                match = re.search(r'-?\d+\.?\d*', answer_str)
                if match:
                    return round(float(match.group()), 2)
            except:
                pass
        
        # Default: return as-is
        return answer_str
    
    def _calculate_confidence(self, state: AgentState, base_confidence: float) -> float:
        """Calculate confidence score using heuristics."""
        confidence = 0.5  # Start with base
        
        # Add for good retrieval
        if state.get("retrieval_score", 0) > 0.5:
            confidence += 0.2
        
        # Add for successful SQL
        if state.get("sql_success"):
            confidence += 0.2
            if state.get("sql_results"):
                confidence += 0.1
        
        # Penalize for repairs
        confidence -= state.get("repair_count", 0) * 0.15
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def run(self, question: str, format_hint: str, question_id: str = "test") -> Dict:
        """
        Run the agent on a single question.
        
        Args:
            question: User's question
            format_hint: Expected output format
            question_id: Question ID for checkpointing
            
        Returns:
            Final state dict
        """
        initial_state = {
            "question": question,
            "format_hint": format_hint,
            "route": "",
            "retrieval_results": [],
            "retrieval_score": 0.0,
            "constraints": "",
            "sql": "",
            "sql_results": [],
            "sql_error": "",
            "sql_success": False,
            "repair_count": 0,
            "final_answer": None,
            "confidence": 0.0,
            "explanation": "",
            "citations": []
        }
        
        config = {"configurable": {"thread_id": question_id}}
        
        # Run the graph
        final_state = self.graph.invoke(initial_state, config)
        
        return final_state
