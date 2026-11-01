from src.llama_rag import generate_answer

answer = generate_answer(
    "What is artificial intelligence?",
    "Artificial intelligence is a field of computer science."
)

print(answer)
