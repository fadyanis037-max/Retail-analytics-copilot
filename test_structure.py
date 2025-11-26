"""
Test script to verify project structure and basic functionality.
Tests without requiring Ollama to be installed.
"""

import os
import sys
import json

# Add agent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_project_structure():
    """Test that all required files and directories exist."""
    print("Testing project structure...")
    
    required_files = [
        "docs/marketing_calendar.md",
        "docs/kpi_definitions.md",
        "docs/catalog.md",
        "docs/product_policy.md",
        "data/northwind.sqlite",
        "sample_questions_hybrid_eval.jsonl",
        "agent/graph_hybrid.py",
        "agent/dspy_signatures.py",
        "agent/rag/retrieval.py",
        "agent/tools/sqlite_tool.py",
        "run_agent_hybrid.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_good = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} MISSING")
            all_good = False
    
    return all_good


def test_document_loading():
    """Test that documents can be loaded and chunked."""
    print("\nTesting document loading...")
    
    try:
        from agent.rag.retrieval import DocumentRetriever
        
        retriever = DocumentRetriever("docs")
        retriever.load_and_index()
        
        print(f"  ✓ Loaded {len(retriever.chunks)} document chunks")
        
        # Test search
        results = retriever.search("return policy beverages", top_k=2)
        print(f"  ✓ Search returned {len(results)} results")
        
        if results:
            print(f"    Top result: {results[0].id} (score: {results[0].score:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_database_connection():
    """Test SQLite database access."""
    print("\nTesting database connection...")
    
    try:
        from agent.tools.sqlite_tool import SQLiteTool
        
        db = SQLiteTool("data/northwind.sqlite")
        
        # Test schema extraction
        schema = db.get_schema()
        print(f"  ✓ Schema extracted ({len(schema)} chars)")
        
        # Test simple query
        success, results, error = db.execute_query("SELECT COUNT(*) as cnt FROM Orders")
        
        if success:
            print(f"  ✓ Query executed: {results[0]['cnt']} orders in database")
        else:
            print(f"  ✗ Query failed: {error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_eval_dataset():
    """Test that eval dataset is properly formatted."""
    print("\nTesting eval dataset...")
    
    try:
        with open("sample_questions_hybrid_eval.jsonl", 'r') as f:
            questions = [json.loads(line) for line in f if line.strip()]
        
        print(f"  ✓ Loaded {len(questions)} questions")
        
        required_fields = ['id', 'question', 'format_hint']
        for q in questions:
            for field in required_fields:
                if field not in q:
                    print(f"  ✗ Missing field '{field}' in question {q.get('id', '?')}")
                    return False
        
        print(f"  ✓ All questions have required fields")
        
        # Show question IDs
        print("\n  Question IDs:")
        for q in questions:
            print(f"    - {q['id']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Retail Analytics Copilot - Structure Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Project Structure", test_project_structure()))
    results.append(("Document Loading", test_document_loading()))
    results.append(("Database Connection", test_database_connection()))
    results.append(("Eval Dataset", test_eval_dataset()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Install Ollama: https://ollama.com")
        print("2. Pull Phi-3.5 model: ollama pull phi3.5:3.8b-mini-instruct-q4_K_M")
        print("3. Run the agent: python run_agent_hybrid.py --batch sample_questions_hybrid_eval.jsonl --out outputs_hybrid.jsonl")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
