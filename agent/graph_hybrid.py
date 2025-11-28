import dspy
from typing import TypedDict, List, Annotated, Dict, Any, Union, Literal
from langgraph.graph import StateGraph, END
import operator
import json

from agent.dspy_signatures import RouterSignature, SQLGeneratorSignature, SynthesizerSignature, PlannerSignature
from agent.rag.retrieval import LocalRetriever
from agent.tools.sqlite_tool import SQLiteTool

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    format_hint: str
    messages: List[str] # Log of steps
    strategy: str
    retrieved_docs: List[Dict]
    constraints: str
    sql_query: str
    sql_result: Dict[str, Any]
    final_answer: Any
    citations: List[str]
    explanation: str
    errors: List[str]
    retry_count: int

# --- Nodes ---

class RetailAgent:
    def __init__(self):
        self.retriever = LocalRetriever()
        self.sqlite_tool = SQLiteTool()
        
        # DSPy Modules
        self.router = dspy.ChainOfThought(RouterSignature)
        self.planner = dspy.ChainOfThought(PlannerSignature)
        
        # Load optimized SQL Generator if exists
        self.sql_generator = dspy.ChainOfThought(SQLGeneratorSignature)
        try:
            self.sql_generator.load("agent/optimized_sql_gen.json")
            print("Loaded optimized SQL Generator.")
        except:
            pass
            
        self.synthesizer = dspy.ChainOfThought(SynthesizerSignature)

    def route_query(self, state: AgentState) -> AgentState:
        """Determines the strategy."""
        pred = self.router(question=state["question"])
        # Normalize strategy
        strategy = pred.strategy.lower().strip()
        if "sql" in strategy and "rag" in strategy:
            strategy = "hybrid"
        elif "sql" in strategy:
            strategy = "sql"
        elif "rag" in strategy:
            strategy = "rag"
        else:
            strategy = "hybrid" # Default
            
        state["strategy"] = strategy
        state["messages"].append(f"Router selected: {strategy}")
        return state

    def retrieve_docs(self, state: AgentState) -> AgentState:
        """Retrieves documents."""
        docs = self.retriever.retrieve(state["question"], k=3)
        state["retrieved_docs"] = docs
        state["messages"].append(f"Retrieved {len(docs)} chunks")
        return state

    def plan_query(self, state: AgentState) -> AgentState:
        """Extracts constraints."""
        context = "\n".join([f"{d['id']}: {d['content']}" for d in state["retrieved_docs"]])
        pred = self.planner(question=state["question"], context=context)
        state["constraints"] = pred.constraints
        state["messages"].append(f"Planned constraints: {pred.constraints}")
        return state

    def generate_sql(self, state: AgentState) -> AgentState:
        """Generates SQL."""
        schema = self.sqlite_tool.get_schema()
        constraints = state.get("constraints", "")
        
        # If retrying, include error context
        question_context = state["question"]
        if state["retry_count"] > 0 and state["errors"]:
            question_context += f"\nPrevious Error: {state['errors'][-1]}"
            
        pred = self.sql_generator(question=question_context, schema=schema, constraints=constraints)
        
        # Clean SQL (remove markdown code blocks if present)
        sql = pred.sql_query.replace("```sql", "").replace("```", "").strip()
        state["sql_query"] = sql
        state["messages"].append(f"Generated SQL: {sql}")
        return state

    def execute_sql(self, state: AgentState) -> AgentState:
        """Executes SQL."""
        result = self.sqlite_tool.execute_sql(state["sql_query"])
        state["sql_result"] = result
        if result["error"]:
            state["errors"].append(result["error"])
            state["messages"].append(f"SQL Execution Error: {result['error']}")
        else:
            state["messages"].append(f"SQL Executed. Rows: {len(result['rows'])}")
        return state

    def synthesize_answer(self, state: AgentState) -> AgentState:
        """Synthesizes the final answer with robust parsing."""
        import re
        import ast
        
        context = "\n".join([f"{d['id']}: {d['content']}" for d in state["retrieved_docs"]])
        sql_res_str = str(state.get("sql_result", {}))
        
        # Truncate SQL result if too long for context window
        if len(sql_res_str) > 2000:
             sql_res_str = sql_res_str[:2000] + "... (truncated)"

        # Try DSPy synthesis with error handling
        try:
            pred = self.synthesizer(
                question=state["question"],
                context=context,
                sql_result=sql_res_str,
                format_hint=state["format_hint"]
            )
            
            final_answer = pred.final_answer
            explanation = pred.explanation
            raw_citations = pred.citations
            
        except Exception as e:
            # Fallback: Manual LLM call and parsing
            state["messages"].append(f"DSPy synthesis failed, using fallback: {str(e)[:100]}")
            
            # Direct LLM call
            lm = dspy.settings.lm
            prompt = f"""Answer this question precisely in the requested format.

Question: {state["question"]}
Format required: {state["format_hint"]}

Context from documents:
{context[:500]}

SQL Result:
{sql_res_str[:500]}

Provide your answer in this exact format:
ANSWER: <your answer matching the format exactly>
EXPLANATION: <1-2 sentence explanation>
CITATIONS: <comma-separated list of tables/docs used>"""

            response = lm(prompt)
            response_text = str(response)
            
            # Parse response
            answer_match = re.search(r'ANSWER:\s*(.+?)(?=EXPLANATION:|$)', response_text, re.DOTALL)
            expl_match = re.search(r'EXPLANATION:\s*(.+?)(?=CITATIONS:|$)', response_text, re.DOTALL)
            cite_match = re.search(r'CITATIONS:\s*(.+?)$', response_text, re.DOTALL)
            
            final_answer = answer_match.group(1).strip() if answer_match else "Unable to determine"
            explanation = expl_match.group(1).strip() if expl_match else "Processed from available data."
            raw_citations = cite_match.group(1).strip() if cite_match else ""
        
        # Parse and type-cast final answer based on format_hint
        format_hint = state["format_hint"].lower()
        
        try:
            if format_hint == "int":
                # Extract integer
                match = re.search(r'\d+', str(final_answer))
                final_answer = int(match.group()) if match else 0
                
            elif format_hint == "float":
                # Extract float
                match = re.search(r'[\d.]+', str(final_answer))
                final_answer = float(match.group()) if match else 0.0
                
            elif "{" in format_hint or "dict" in format_hint:
                # Try to parse as dict
                if isinstance(final_answer, str):
                    try:
                        final_answer = ast.literal_eval(final_answer)
                    except:
                        # Try JSON
                        import json
                        try:
                            final_answer = json.loads(final_answer)
                        except:
                            # Extract key-value pairs manually
                            final_answer = {}
                            
            elif "list" in format_hint:
                # Try to parse as list
                if isinstance(final_answer, str):
                    try:
                        final_answer = ast.literal_eval(final_answer)
                    except:
                        import json
                        try:
                            final_answer = json.loads(final_answer)
                        except:
                            final_answer = []
        except Exception as parse_err:
            state["messages"].append(f"Type conversion warning: {str(parse_err)[:100]}")
        
        state["final_answer"] = final_answer
        state["explanation"] = explanation[:200]  # Limit explanation length
        
        # Clean citations
        if isinstance(raw_citations, str):
            # Try to parse string list
            try:
                raw_citations = ast.literal_eval(raw_citations)
            except:
                # Split by common delimiters
                raw_citations = [c.strip() for c in re.split(r'[,;\n]', raw_citations) if c.strip()]
        
        # Add SQL tables to citations if SQL was used
        if state.get("sql_query") and not state.get("sql_result", {}).get("error"):
            # Extract table names from SQL
            sql = state["sql_query"].lower()
            common_tables = ["orders", "order_items", "products", "customers", "categories", "suppliers"]
            for table in common_tables:
                if table in sql and table not in [str(c).lower() for c in raw_citations]:
                    raw_citations.append(table)
        
        # Add doc chunk IDs to citations
        for doc in state["retrieved_docs"]:
            doc_id = doc.get("id", doc.get("full_id", ""))
            if doc_id and doc_id not in raw_citations:
                raw_citations.append(doc_id)
        
        state["citations"] = list(set(raw_citations))  # Remove duplicates
        state["messages"].append("Synthesized answer with fallback parsing")
        return state

    def repair_node(self, state: AgentState) -> AgentState:
        """Increments retry count and prepares for repair."""
        state["retry_count"] += 1
        state["messages"].append(f"Triggering repair. Retry count: {state['retry_count']}")
        return state

    def check_repair(self, state: AgentState) -> Literal["repair", "end"]:
        """Decides whether to repair or end."""
        # Check 1: SQL Error
        if state.get("sql_result") and state["sql_result"].get("error"):
            if state["retry_count"] < 2:
                return "repair"
        
        return "end"

# --- Graph Construction ---

def build_graph():
    agent = RetailAgent()
    workflow = StateGraph(AgentState)

    workflow.add_node("router", agent.route_query)
    workflow.add_node("retriever", agent.retrieve_docs)
    workflow.add_node("planner", agent.plan_query)
    workflow.add_node("sql_generator", agent.generate_sql)
    workflow.add_node("executor", agent.execute_sql)
    workflow.add_node("repair_node", agent.repair_node)
    workflow.add_node("synthesizer_node", agent.synthesize_answer)

    # Edges
    workflow.set_entry_point("router")
    
    # Router logic
    def route_decision(state):
        if state["strategy"] == "rag":
            return "retriever_only"
        elif state["strategy"] == "sql":
            return "sql_only"
        else:
            return "hybrid"

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retriever_only": "retriever",
            "sql_only": "sql_generator",
            "hybrid": "retriever"
        }
    )

    # Hybrid/RAG path
    workflow.add_edge("retriever", "planner")
    
    def plan_decision(state):
        if state["strategy"] == "rag":
            return "synthesizer_node"
        else:
            return "sql_generator"

    workflow.add_conditional_edges(
        "planner",
        plan_decision,
        {
            "synthesizer_node": "synthesizer_node",
            "sql_generator": "sql_generator"
        }
    )

    # SQL path
    workflow.add_edge("sql_generator", "executor")
    
    def execution_decision(state):
        if state["sql_result"].get("error") and state["retry_count"] < 2:
            return "repair_sql"
        return "synthesizer_node"

    workflow.add_conditional_edges(
        "executor",
        execution_decision,
        {
            "repair_sql": "repair_node", 
            "synthesizer_node": "synthesizer_node"
        }
    )
    
    workflow.add_edge("repair_node", "sql_generator")

    # Synthesizer to End
    workflow.add_edge("synthesizer_node", END)

    return workflow.compile()
