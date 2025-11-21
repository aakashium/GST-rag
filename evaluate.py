import os
import json
import time
from datetime import datetime
from rag_engine import get_rag_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def evaluate_rag():
    print("Initializing RAG Chain and Evaluator...")
    qa_chain, retriever = get_rag_chain()
    eval_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

    test_questions = [
        "What is the penalty for not filing GST returns?",
        "Explain the composition scheme.",
        "What are the rules for input tax credit?",
        "How is GST calculated on inter-state supply?"
    ]

    print("\n--- Starting Evaluation ---\n")
    
    # Store results
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": "gemini-2.0-flash-exp",
        "evaluations": []
    }

    for question in test_questions:
        print(f"Question: {question}")
        
        # Add delay to avoid hitting rate limits (60 seconds between requests)
        if test_questions.index(question) > 0:
            print("Waiting 60 seconds to avoid rate limits...")
            time.sleep(60)
        
        # Get RAG response
        answer = qa_chain.invoke(question)
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        print(f"Answer: {answer}\n")

        # Evaluation Prompt (LLM-as-a-judge)
        eval_prompt = f"""
        You are an evaluator for a RAG system.
        
        Question: {question}
        Generated Answer: {answer}
        Retrieved Context: {context}
        
        Task:
        1. Rate "Answer Relevance" (1-5): Does the answer directly address the question?
        2. Rate "Context Precision" (1-5): Is the retrieved context relevant to the question?
        3. Provide a brief explanation.
        
        Output Format:
        Relevance: [Score]
        Precision: [Score]
        Explanation: [Text]
        """
        
        # Add delay before evaluation call (10 seconds)
        time.sleep(10)
        
        eval_result = eval_llm.invoke(eval_prompt).content
        print(f"Evaluation:\n{eval_result}\n")
        print("-" * 50)
        
        # Store result
        results["evaluations"].append({
            "question": question,
            "answer": answer,
            "context_snippets": [doc.page_content[:200] + "..." for doc in docs],
            "evaluation": eval_result
        })
    
    # Save results to JSON file
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"evaluation_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Evaluation results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    evaluate_rag()

