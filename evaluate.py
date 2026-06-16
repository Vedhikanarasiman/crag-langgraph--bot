import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.graph.graph import build_graph

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ─────────────────────────────────────────
# TEST DATASET
# ─────────────────────────────────────────
test_data = [
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": "The attention mechanism allows the model to weigh importance of different words using Query, Key and Value matrices with scaled dot-product attention formula."
    },
    {
        "question": "What is Retrieval Augmented Generation?",
        "ground_truth": "RAG combines retrieval of relevant documents from an external knowledge base with language model generation to reduce hallucinations and improve factual accuracy."
    },
    {
        "question": "What is the ReAct framework?",
        "ground_truth": "ReAct combines reasoning and acting in language models by interleaving chain-of-thought reasoning with action steps iteratively."
    },
    {
        "question": "What is SELF-RAG?",
        "ground_truth": "SELF-RAG trains language models to adaptively retrieve passages on demand and reflect on them using special critique tokens to improve output quality."
    },
    {
        "question": "What is chain of thought prompting?",
        "ground_truth": "Chain of thought prompting encourages LLMs to break down complex reasoning into intermediate steps, improving performance on arithmetic and reasoning tasks."
    }
]

# ─────────────────────────────────────────
# METRIC 1: FAITHFULNESS
# Is the answer grounded in retrieved chunks?
# ─────────────────────────────────────────
def evaluate_faithfulness(answer: str, contexts: list[str]) -> float:
    context_text = "\n".join(contexts)

    prompt = f"""You are evaluating if an answer is faithful to the given context.

Context:
{context_text}

Answer:
{answer}

Score the faithfulness from 0.0 to 1.0:
- 1.0 = Every claim in the answer is directly supported by the context
- 0.5 = Some claims are supported, some are not
- 0.0 = Answer is not supported by the context at all

Reply with ONLY a number between 0.0 and 1.0"""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5

# ─────────────────────────────────────────
# METRIC 2: ANSWER RELEVANCY
# Does the answer actually address the question?
# ─────────────────────────────────────────
def evaluate_answer_relevancy(question: str, answer: str) -> float:
    prompt = f"""You are evaluating if an answer is relevant to the question.

Question: {question}
Answer: {answer}

Score the relevancy from 0.0 to 1.0:
- 1.0 = Answer directly and completely addresses the question
- 0.5 = Answer partially addresses the question
- 0.0 = Answer is completely irrelevant to the question

Reply with ONLY a number between 0.0 and 1.0"""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5

# ─────────────────────────────────────────
# METRIC 3: CONTEXT RELEVANCY
# Were the right chunks retrieved?
# ─────────────────────────────────────────
def evaluate_context_relevancy(question: str, contexts: list[str]) -> float:
    context_text = "\n---\n".join(contexts)

    prompt = f"""You are evaluating if the retrieved context is relevant to the question.

Question: {question}

Retrieved Context:
{context_text}

Score the context relevancy from 0.0 to 1.0:
- 1.0 = All retrieved chunks are highly relevant to the question
- 0.5 = Some chunks are relevant, some are not
- 0.0 = Retrieved chunks are completely irrelevant

Reply with ONLY a number between 0.0 and 1.0"""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5

# ─────────────────────────────────────────
# METRIC 4: ANSWER CORRECTNESS
# Does the answer match the ground truth?
# ─────────────────────────────────────────
def evaluate_correctness(answer: str, ground_truth: str) -> float:
    prompt = f"""You are evaluating if an answer matches the ground truth.

Ground Truth: {ground_truth}
Generated Answer: {answer}

Score the correctness from 0.0 to 1.0:
- 1.0 = Answer conveys the same information as ground truth
- 0.5 = Answer partially matches the ground truth
- 0.0 = Answer contradicts or misses the ground truth completely

Reply with ONLY a number between 0.0 and 1.0"""

    response = llm.invoke(prompt)
    try:
        score = float(response.content.strip())
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5

# ─────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ─────────────────────────────────────────
def run_evaluation():
    print("=== ResearchMind Custom Evaluation ===\n")

    app = build_graph()
    config = {"configurable": {"thread_id": "eval-session"}}

    results = []

    for i, item in enumerate(test_data):
        print(f"\nQuestion {i+1}/5: {item['question']}")
        print("-" * 50)

        try:
            # Run through pipeline
            result = app.invoke(
                {
                    "question": item["question"],
                    "chat_history": [],
                    "retry_count": 0
                },
                config=config
            )

            answer = result["answer"]
            contexts = [
                chunk["text"]
                for chunk in result.get("filtered_chunks", [])
            ]

            print(f"Answer: {answer[:150]}...")

            # Score each metric
            print("Scoring...")
            faithfulness = evaluate_faithfulness(answer, contexts)
            relevancy = evaluate_answer_relevancy(item["question"], answer)
            context_rel = evaluate_context_relevancy(item["question"], contexts)
            correctness = evaluate_correctness(answer, item["ground_truth"])

            result_item = {
                "question": item["question"],
                "answer": answer,
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
                "context_relevancy": context_rel,
                "correctness": correctness,
                "overall": (faithfulness + relevancy + context_rel + correctness) / 4
            }

            results.append(result_item)

            print(f"  Faithfulness:       {faithfulness:.2f}")
            print(f"  Answer Relevancy:   {relevancy:.2f}")
            print(f"  Context Relevancy:  {context_rel:.2f}")
            print(f"  Correctness:        {correctness:.2f}")
            print(f"  Overall:            {result_item['overall']:.2f}")

        except Exception as e:
            print(f"Error: {e}")
            continue

    # ─────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────
    if results:
        print("\n" + "=" * 50)
        print("=== FINAL EVALUATION SUMMARY ===")
        print("=" * 50)

        avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
        avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
        avg_context = sum(r["context_relevancy"] for r in results) / len(results)
        avg_correctness = sum(r["correctness"] for r in results) / len(results)
        avg_overall = sum(r["overall"] for r in results) / len(results)

        print(f"\nAverage Faithfulness:      {avg_faithfulness:.2f}")
        print(f"Average Answer Relevancy:  {avg_relevancy:.2f}")
        print(f"Average Context Relevancy: {avg_context:.2f}")
        print(f"Average Correctness:       {avg_correctness:.2f}")
        print(f"\nOverall RAG Score:         {avg_overall:.2f} / 1.00")

        print("\nGrade:")
        if avg_overall >= 0.80:
            print("✅ Excellent — Pipeline is performing very well!")
        elif avg_overall >= 0.65:
            print("✅ Good — Pipeline is solid with room to improve")
        elif avg_overall >= 0.50:
            print("⚠️ Fair — Pipeline needs some tuning")
        else:
            print("❌ Poor — Pipeline needs significant improvement")

        # Save results to JSON
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nDetailed results saved to eval_results.json")

if __name__ == "__main__":
    run_evaluation()