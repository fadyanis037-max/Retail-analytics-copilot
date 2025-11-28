import click
import dspy
import json
import os
from typing import List, Dict
from agent.graph_hybrid import build_graph, AgentState

# Configure DSPy
def setup_dspy():
    # Assuming Ollama is running locally on default port
    lm = dspy.LM(model='ollama/phi3.5:3.8b-mini-instruct-q4_K_M', api_base='http://127.0.0.1:11434', api_key='')
    dspy.settings.configure(lm=lm)

@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch, out):
    """Run the Retail Analytics Copilot."""
    setup_dspy()
    
    # Load questions
    questions = []
    with open(batch, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    app = build_graph()
    results = []
    
    print(f"Processing {len(questions)} questions...")
    
    for q_item in questions:
        q_id = q_item['id']
        question = q_item['question']
        format_hint = q_item['format_hint']
        
        print(f"Running: {q_id}")
        
        initial_state = AgentState(
            question=question,
            format_hint=format_hint,
            messages=[],
            strategy="",
            retrieved_docs=[],
            constraints="",
            sql_query="",
            sql_result={},
            final_answer=None,
            citations=[],
            explanation="",
            errors=[],
            retry_count=0
        )
        
        final_state = app.invoke(initial_state)
        
        output = {
            "id": q_id,
            "final_answer": final_state.get("final_answer"),
            "sql": final_state.get("sql_query", ""),
            "confidence": 0.8, # Placeholder confidence
            "explanation": final_state.get("explanation", ""),
            "citations": final_state.get("citations", [])
        }
        
        results.append(output)
        
        # Print trace for debugging
        # for msg in final_state['messages']:
        #     print(f"  - {msg}")

    # Write outputs
    with open(out, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Done. Results written to {out}")

if __name__ == '__main__':
    main()
