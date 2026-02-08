from chat_memory import (
    Message,
    ImportanceCalculator,
    Vectorizer,
    VectorIndex,
    EvictionStrategy,
    MemoryStore,
    Retriever
)


# =============================
# Demo
# =============================

def main():

    # Initialize components
    importance_calculator = ImportanceCalculator()
    vectorizer = Vectorizer()
    vector_index = VectorIndex()
    eviction_strategy = EvictionStrategy()

    memory = MemoryStore(
        max_size=10,
        vectorizer=vectorizer,
        vector_index=vector_index,
        eviction_strategy=eviction_strategy,
        importance_calculator=importance_calculator
    )

    retriever = Retriever(vectorizer, vector_index)

    print("\n=== Adding Chats to Memory ===\n")

    chats = [
        "Explain neural networks",
        "What is deep learning?",
        "I like pizza",
        "Machine learning basics",
        "Tell me about AI models",
        "How does backpropagation work?",
        "Explain gradient descent in simple terms",
        "Difference between supervised and unsupervised learning",
        "What is reinforcement learning?",
        "Neural networks vs traditional algorithms",
        "How transformers work in NLP",
        "Explain large language models",
        "What is overfitting in machine learning?",
        "Tips for studying AI concepts",
        "Hello, how are you today?"
    ]

    for text in chats:
        memory.add_message(Message(text=text))

    print("Chats successfully stored.\n")

    print("Current Memory:")
    for msg in memory.messages:
        print(f"• {msg.text}")

    query = "Explain AI and neural networks"

    print("\n=== Querying System ===")
    print(f"Query: {query}")

    results = retriever.retrieve(query)

    print("\n=== Top Five Relevant Chats ===")

    for i, msg in enumerate(results, 1):
        print(f"{i}. {msg.text}")


if __name__ == "__main__":
    main()
