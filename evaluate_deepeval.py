import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_groq import ChatGroq
from src.graph.graph import build_graph

load_dotenv()

# ─────────────────────────────────────────
# WRAP GROQ AS DEEPEVAL CUSTOM LLM
# DeepEval needs its own LLM wrapper
# ─────────────────────────────────────────
class GroqDeepEvalLLM(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return "llama-3.3-70b-versatile"

# ─────────────────────────────────────────
# TEST DATASET
# ─────────────────────────────────────────
test_data = [
    {
        "question": "How does multi-head attention work in transformers?",
        "ground_truth": "Multi-head attention runs multiple attention functions in parallel on different learned projections of queries, keys and values, then concatenates and projects the results allowing the model to jointly attend to information from different representation subspaces."
    },
    {
        "question": "What is the difference between dense and sparse retrieval?",
        "ground_truth": "Dense retrieval uses neural embeddings to represent documents and queries as dense vectors and finds similar documents using vector similarity. Sparse retrieval uses keyword matching with term frequency weights like BM25. Dense retrieval captures semantic meaning while sparse retrieval is exact term based."
    }
]
# test_data = [
#     {
#         "question": "What is the attention mechanism in transformers?",
#         "ground_truth": "The attention mechanism allows the model to weigh importance of different words using Query, Key and Value matrices with scaled dot-product attention."
#     },
#     {
#         "question": "What is Retrieval Augmented Generation?",
#         "ground_truth": "RAG combines retrieval of relevant documents from an external knowledge base with language model generation to reduce hallucinations and improve factual accuracy."
#     },
#     {
#         "question": "What is the ReAct framework?",
#         "ground_truth": "ReAct combines reasoning and acting in language models by interleaving chain-of-thought reasoning with action steps iteratively."
#     },
#     {
#         "question": "What is SELF-RAG?",
#         "ground_truth": "SELF-RAG trains language models to adaptively retrieve passages and reflect on them using special critique tokens to improve output quality."
#     },
#     {
#         "question": "What is chain of thought prompting?",
#         "ground_truth": "Chain of thought prompting encourages LLMs to break down complex reasoning into intermediate steps improving performance on reasoning tasks."
#     }
# ]

# ─────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────
def run_deepeval():
    print("=== ResearchMind DeepEval Evaluation ===\n")

    # Initialize custom LLM
    groq_llm = GroqDeepEvalLLM()

    # Initialize metrics
    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model=groq_llm,
        include_reason=True
    )
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model=groq_llm,
        include_reason=True
    )
    contextual_precision = ContextualPrecisionMetric(
        threshold=0.7,
        model=groq_llm,
        include_reason=True
    )
    contextual_recall = ContextualRecallMetric(
        threshold=0.7,
        model=groq_llm,
        include_reason=True
    )

    # Build graph
    app = build_graph()
    config = {"configurable": {"thread_id": "deepeval-session"}}

    test_cases = []

    for i, item in enumerate(test_data):
        print(f"Running question {i+1}/5: {item['question'][:50]}...")

        try:
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

            # Create DeepEval test case
            test_case = LLMTestCase(
                input=item["question"],
                actual_output=answer,
                expected_output=item["ground_truth"],
                retrieval_context=contexts
            )

            test_cases.append(test_case)
            print(f"✅ Question {i+1} done")

        except Exception as e:
            print(f"❌ Error on question {i+1}: {e}")
            continue

    # Run DeepEval
    print("\n=== Running DeepEval Metrics ===\n")

    results = evaluate(
        test_cases=test_cases,
        metrics=[
            faithfulness,
            answer_relevancy,
            contextual_precision,
            contextual_recall
        ]
    )

    # Print summary
    print("\n=== DEEPEVAL RESULTS SUMMARY ===\n")
    for test_result in results.test_results:
        print(f"Question: {test_result.input[:60]}")
        for metric_data in test_result.metrics_data:
            status = "✅" if metric_data.success else "❌"
            print(f"  {status} {metric_data.name}: {metric_data.score:.2f}")
            if metric_data.reason:
                print(f"     Reason: {metric_data.reason[:100]}")
        print()

if __name__ == "__main__":
    run_deepeval()