# Chat Memory Retrieval System

A Python implementation of a hybrid semantic memory system that stores conversational history and retrieves the most relevant past interactions using embeddings, adaptive memory management, and hybrid relevance scoring.

## 🚀 Quick Start

```bash

pip install -r requirements.txt
python demo.py

```



## Overview

This project implements a structured conversational memory system that stores recent chat interactions and retrieves the top five most relevant past messages based on an input query.

Rather than relying on simple keyword matching, the system models memory using semantic understanding, contextual importance, and temporal relevance. The goal is to simulate a more human-like recall process where meaningful, recent, and frequently accessed information is prioritized.

The implementation follows a modular architecture separating storage, indexing, retrieval, and memory management for clarity, scalability, and maintainability.

---

# 🎯 Problem Approach

The system was designed around a core question:

> How can chat history behave more like human memory instead of a static database?

Human recall is influenced by:

* Meaning (semantic similarity)
* Importance (significance of information)
* Recency (temporal context)
* Usage (frequent recall reinforces memory)

This project combines these signals into a hybrid retrieval model.

---

# 🧱 Architecture Overview

The system is divided into clearly defined components:

```
Message
    ↓
MemoryStore
    ↓
Vectorizer → VectorIndex
    ↓
Retriever
    ↓
EvictionStrategy
```

Each component has a single responsibility, ensuring clarity and extensibility.

---

## 1️⃣ Message (@dataclass)

Represents a single chat entry.

Fields include:

* text content
* sender role (user/system/assistant)
* timestamp (automatic)
* importance score
* embedding vector
* access_count (usage tracking)

Design rationale:

* Using a dataclass ensures structured, readable, and extensible data representation.

---

## 2️⃣ ImportanceCalculator

Assigns an importance score using lightweight heuristics:

* Message length
* Presence of question indicators
* Sender role weighting

Purpose:

* Introduce contextual weighting beyond semantic similarity.
* Influence both retrieval ranking and memory retention.

---

## 3️⃣ Vectorizer (State-Aware)

Responsible for generating semantic embeddings using Sentence Transformers.

Key properties:

* Embeddings capture contextual meaning.
* Supports batch processing.
* Attaches embedding directly to message objects.

Why state-aware?

Embedding generation is centralized to ensure consistency and reduce redundant computation.

---

## 4️⃣ VectorIndex (Real-Time Updating)

Acts as the semantic search layer.

Responsibilities:

* Maintain embedding list separate from storage.
* Perform cosine similarity searches efficiently.
* Return candidate messages for ranking.

Design decision:

Separating indexing from storage allows future replacement with scalable vector databases without modifying memory logic.

---

## 5️⃣ MemoryStore

Central orchestrator managing message lifecycle.

Responsibilities:

* Add new messages.
* Automatically compute importance if missing.
* Generate embeddings.
* Update VectorIndex immediately.
* Enforce memory size limits using adaptive eviction.

This component acts as the system’s “source of truth.”

---

## 6️⃣ EvictionStrategy (Adaptive Memory)

Maintains bounded memory size.

When capacity is exceeded:

Messages are evaluated using a survival score:

```
Survival Score =
    importance
  + recency
  + usage frequency
```

Lowest scoring messages are removed.

Rationale:

Simulates realistic forgetting patterns where irrelevant or unused information fades.

---

## 7️⃣ Retriever (Hybrid Scoring)

Handles query-based retrieval.

### Retrieval Flow

1. Convert query into embedding.
2. Retrieve semantic candidate pool.
3. Apply hybrid scoring:

```
Final Score =
    70% semantic similarity
    20% importance
    10% recency
```

4. Return top five messages.
5. Update access_count.

Design reasoning:

Semantic similarity dominates ranking, while importance and recency provide contextual refinement.

---

# 🔎 Relevance Strategy Explained

The system uses a hybrid ranking approach:

### Semantic Similarity

Embeddings generated using Sentence Transformers allow matching based on meaning rather than exact keywords.

Example:

* “deep learning” and “neural networks” are identified as related even without shared words.

---

### Importance Weighting

Messages deemed significant receive ranking preference, ensuring key conversations remain accessible.

---

### Recency (Linear Decay)

Recent messages are slightly favored, reflecting conversational context dynamics.

---

### Usage Tracking

Frequently retrieved messages influence memory retention through adaptive eviction.

---

# 🤖 Why Sentence Transformers?

Sentence Transformers were selected because they:

* Capture semantic meaning effectively.
* Enable contextual similarity beyond keyword overlap.
* Are lightweight and easy to integrate.
* Represent industry-standard practice for semantic retrieval systems.

Alternative approaches such as keyword matching were avoided because they fail to capture paraphrases or conceptual similarity.

---

# 🔄 System Flow

## Adding Chats

```
Create Message
→ Calculate Importance
→ Generate Embedding
→ Store in Memory
→ Update Vector Index
→ Apply Eviction (if needed)
```

---

## Querying

```
User Query
→ Convert to Embedding
→ Retrieve Semantic Candidates
→ Apply Hybrid Scoring
→ Return Top Five Messages
```

---

# 📊 Demonstration Example

## Adding Chats

```
memory.add_message(Message("Explain neural networks"))
memory.add_message(Message("Machine learning basics"))
memory.add_message(Message("I like pizza"))
```

Explanation:

Messages are embedded automatically and stored with adaptive memory management.

---

## Querying

```
results = retriever.retrieve("Explain AI concepts")
```

Explanation:

The query is converted to a semantic representation, and relevant past chats are ranked using hybrid scoring.

---

## Output (Example)

```
1. Explain neural networks
2. Machine learning basics
3. Tell me about AI models
4. Deep learning explanation
5. Neural networks vs traditional algorithms
```

---

# ✅ Key Design Decisions

* Hybrid relevance model for balanced retrieval.
* Clear separation between storage and indexing.
* Adaptive memory pruning.
* Modular architecture enabling scalability.

---


## Design Tradeoffs

Several design decisions were made to balance simplicity, clarity, and scalability:

### Hybrid Scoring Instead of Complex Models

A weighted hybrid approach combining semantic similarity, importance, and recency was chosen instead of machine learning ranking models. This keeps the system interpretable, lightweight, and easy to maintain while still providing meaningful relevance.

---

### Separate Storage and Index Layers

MemoryStore and VectorIndex are intentionally separated:

* MemoryStore manages lifecycle and eviction.
* VectorIndex handles semantic search.

Although this creates duplication of references, it enables clean separation of concerns and allows the indexing mechanism to be replaced independently in the future.

---

### Linear Recency Decay

Linear decay was selected over exponential or probabilistic decay because:

* It is predictable.
* Easy to reason about.
* Sufficient for assignment-scale memory behavior.

---

### Sentence Transformers vs Keyword Matching

Semantic embeddings were preferred over keyword overlap methods because:

* They capture contextual meaning.
* Handle paraphrases naturally.
* Improve retrieval quality without significant complexity.

---


# 🚀 Conclusion

This project demonstrates a scalable approach to conversational memory retrieval by combining semantic embeddings, contextual weighting, and adaptive memory management. The design prioritizes clarity, maintainability, and realistic memory behavior while remaining accessible and easy to extend.

