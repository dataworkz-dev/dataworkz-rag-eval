from groq_openai import GroqOpenAI
from continuous_eval.metrics.retrieval import ContextPrecision


llm = GroqOpenAI(
    api_key="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",
    endpoint="https://api.groq.com/openai/v1",
    model="deepseek-r1-distill-llama-70b",
)

datum = {
    "question": "What is the capital of India?",
    "retrieved_context": [
        "Paris is the capital of France and also the largest city in the country.",
        "Bangalore is a major city in India.",
    ],
}

metric = ContextPrecision()
print(metric(**datum))
