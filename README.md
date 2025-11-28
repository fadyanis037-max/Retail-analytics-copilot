# Retail Analytics Copilot

A local, free AI agent for retail analytics using LangGraph and DSPy.

## Graph Design
- **Hybrid Router**: Classifies questions as SQL, RAG, or Hybrid to select the optimal path.
- **RAG & SQL Paths**: Parallel execution paths for document retrieval and SQL generation.
- **Planner & Synthesizer**: Extracts constraints from docs to guide SQL, and synthesizes final answers with citations.
- **Repair Loop**: Automatically retries SQL generation on errors (up to 2 times).

## DSPy Optimization
- **Module**: `SQLGenerator`
- **Optimization**: Used `BootstrapFewShot` to improve SQL generation accuracy.
- **Metric**: Valid SQL execution rate.
- **Results**: (To be updated after evaluation)

## Trade-offs & Assumptions
- **CostOfGoods**: Approximated as 0.7 * UnitPrice where missing, as per instructions.
- **Local Model**: Relies on `phi3.5:3.8b-mini-instruct-q4_K_M` via Ollama. Performance depends on local hardware.
- **Schema**: Simplified schema views used for easier SQL generation.

## Usage
1. Ensure Ollama is running and model is pulled:
   ```bash
   ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
   ```
2. Run the agent:
   ```bash
   python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
   ```
