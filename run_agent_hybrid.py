"""
CLI entrypoint for the Retail Analytics Copilot.
Runs batch evaluation on questions from a JSONL file.
"""

import os
import sys
import json
import click
from rich.console import Console
from rich.progress import track

# Add agent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.graph_hybrid import HybridAgent


console = Console()


@click.command()
@click.option(
    '--batch',
    required=True,
    type=click.Path(exists=True),
    help='Path to JSONL file with questions'
)
@click.option(
    '--out',
    required=True,
    type=click.Path(),
    help='Output path for results JSONL'
)
def main(batch: str, out: str):
    """
    Run Retail Analytics Copilot on batch questions.
    
    Example:
        python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl
    """
    console.print("[bold blue]Retail Analytics Copilot[/bold blue]")
    console.print("Initializing agent...\n")
    
    # Initialize agent
    project_root = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(project_root, "docs")
    db_path = os.path.join(project_root, "data", "northwind.sqlite")
    
    if not os.path.exists(docs_dir):
        console.print(f"[red]Error: docs directory not found at {docs_dir}[/red]")
        sys.exit(1)
    
    if not os.path.exists(db_path):
        console.print(f"[red]Error: database not found at {db_path}[/red]")
        sys.exit(1)
    
    try:
        agent = HybridAgent(docs_dir=docs_dir, db_path=db_path)
        console.print("[green]✓[/green] Agent initialized\n")
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        sys.exit(1)
    
    # Load questions
    questions = []
    with open(batch, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    console.print(f"Loaded {len(questions)} questions\n")
    
    # Process each question
    results = []
    for q in track(questions, description="Processing questions..."):
        console.print(f"\n[cyan]Q:[/cyan] {q['question'][:80]}...")
        
        try:
            # Run agent
            state = agent.run(
                question=q['question'],
                format_hint=q['format_hint'],
                question_id=q['id']
            )
            
            # Build output following contract
            output = {
                "id": q['id'],
                "final_answer": state.get("final_answer"),
                "sql": state.get("sql", ""),
                "confidence": round(state.get("confidence", 0.0), 2),
                "explanation": state.get("explanation", ""),
                "citations": state.get("citations", [])
            }
            
            results.append(output)
            
            console.print(f"[green]✓[/green] Answer: {output['final_answer']}")
            console.print(f"  Confidence: {output['confidence']}")
            console.print(f"  Citations: {len(output['citations'])} items")
            
        except Exception as e:
            console.print(f"[red]Error processing question {q['id']}: {e}[/red]")
            # Still add a result with error info
            results.append({
                "id": q['id'],
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            })
    
    # Write results
    with open(out, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    console.print(f"\n[bold green]✓ Done![/bold green]")
    console.print(f"Results written to: {out}")


if __name__ == '__main__':
    main()
