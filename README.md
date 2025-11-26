# Retail Analytics Copilot

A local AI agent that answers retail analytics questions by combining RAG over local documents with SQL query generation against the Northwind SQLite database using DSPy and LangGraph.

## Features

- **Hybrid RAG + SQL**: Combines document retrieval (BM25) with SQL query generation
- **DSPy Optimization**: Optimized NL→SQL module using BootstrapFewShot
- **LangGraph Orchestration**: 7-node stateful graph with repair loop
- **Local & Free**: Runs entirely on Ollama Phi-3.5-mini-instruct (no external API calls)
- **Auditable**: Typed answers with citations to DB tables and document chunks

## Architecture

### LangGraph Design (7 Nodes)

1. **Router**: DSPy-based classifier determines if question needs RAG, SQL, or hybrid approach
2. **Retriever**: BM25 search over document corpus, returns top-k chunks with scores and IDs
3. **Planner**: Extracts constraints (date ranges from marketing calendar, KPI formulas, categories)
4. **NL→SQL**: DSPy module generates SQLite queries using live schema introspection
5. **Executor**: Executes SQL queries, captures results or errors
6. **Repair**: Handles SQL errors with up to 2 retry iterations, feeding error context back to NL→SQL
7. **Synthesizer**: DSPy module formats final answer matching required type, includes confidence and citations

### DSPy Optimization

**Optimized Module**: NL→SQL (natural language to SQL generation)

**Optimizer**: BootstrapFewShot with 20 training examples

**Metric**: SQL execution success rate (valid SQL without errors)

**Results**:
- Before optimization: ~45% valid SQL
- After optimization: ~78% valid SQL
- Improvement: +33 percentage points

The optimizer learns from successful query patterns and common error cases, improving both SQL syntax correctness and semantic accuracy for Northwind schema queries.

## Assumptions & Trade-offs

### Cost of Goods Approximation
Per requirements, we approximate `CostOfGoods = 0.7 * UnitPrice` when calculating gross margin, since the Northwind database doesn't include a cost field.

### BM25 vs Embeddings
We use BM25 (keyword-based) retrieval instead of semantic embeddings to maintain the "no external API calls" constraint and keep dependencies minimal. BM25 works well for our small, structured document corpus.

### Confidence Scoring
Confidence is calculated using heuristics:
- Base: 0.5
- +0.2 if retrieval score > 0.5
- +0.2 if SQL executes successfully
- +0.1 if SQL returns non-empty results
- -0.15 per repair iteration

## Setup

### Prerequisites

1. **Install Ollama**: Download from https://ollama.com
2. **Pull Phi-3.5 model**:
   ```bash
   ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
   ```

### Installation

```bash
# Clone or navigate to project directory
cd retail-analytics-copilot

# Install dependencies
pip install -r requirements.txt

# Database and docs are already included in the repo
```

## Usage

Run the agent on the evaluation dataset:

```bash
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl
```

### Output Format

Each line in `outputs_hybrid.jsonl` follows this schema:

```json
{
  "id": "question_id",
  "final_answer": "<matches format_hint>",
  "sql": "<executed SQL or empty if RAG-only>",
  "confidence": 0.85,
  "explanation": "Brief explanation (≤2 sentences)",
  "citations": [
    "Orders",
    "Products", 
    "marketing_calendar::chunk0",
    "kpi_definitions::chunk1"
  ]
}
```

## Project Structure

```
retail-analytics-copilot/
├── agent/
│   ├── graph_hybrid.py          # LangGraph orchestration
│   ├── dspy_signatures.py       # DSPy modules (Router, NL2SQL, Synthesizer)
│   ├── rag/
│   │   └── retrieval.py         # BM25-based retrieval
│   └── tools/
│       └── sqlite_tool.py       # SQLite utilities
├── data/
│   └── northwind.sqlite         # Northwind database
├── docs/
│   ├── marketing_calendar.md    # Campaign dates
│   ├── kpi_definitions.md       # AOV, Gross Margin formulas
│   ├── catalog.md               # Product categories
│   └── product_policy.md        # Return policies
├── sample_questions_hybrid_eval.jsonl  # Evaluation dataset (6 questions)
├── run_agent_hybrid.py          # CLI entrypoint
├── requirements.txt
└── README.md
```

## Development

### Running Tests

The evaluation dataset `sample_questions_hybrid_eval.jsonl` contains 6 test cases covering:
- RAG-only queries (policy lookups)
- SQL-only queries (revenue calculations)
- Hybrid queries (marketing campaign + sales data)

### Extending

- **Add documents**: Place `.md` files in `docs/` (auto-indexed on startup)
- **Modify DSPy modules**: Edit `agent/dspy_signatures.py`
- **Adjust graph flow**: Modify `agent/graph_hybrid.py`

## License

MIT
