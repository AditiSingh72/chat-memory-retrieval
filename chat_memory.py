from dataclasses import dataclass, field
from typing import Optional, List
import time
import uuid
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Message:
    """
    Represents a single chat message stored in memory.

    """

    text: str
    role: str = "user"
    importance: float = 0.0

    # auto-generated fields

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # added later by vectorizer

    embedding: Optional[List[float]] = None

    # updated during retrieval

    access_count: int = 0


# =============================
# Importance Calculator
# =============================

class ImportanceCalculator:
    """
    Calculates importance score for a message using simple heuristics.

    """

    def calculate(self, text: str, role: str = "user") -> float:

        score = 0.0

        # Length heuristic 
        length_score = min(len(text) / 100, 1.0)
        score += length_score

        # questions often indicate intent
        if "?" in text:
            score += 0.2

        # Role-based boost
        if role == "system":
            score += 0.3

        # Clamp between 0 and 1
        return min(score, 1.0)
    


    

# =============================
# Vectorizer
# =============================



class Vectorizer:
    """
    Converts text into semantic embeddings.
    Supports single and batch encoding.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, message):
        """
        Encode a single message and attach embedding.

        """
        embedding = self.model.encode(message.text)

        message.embedding = embedding

        return embedding

    def encode_batch(self, messages):
        """
        Encode multiple messages at once.

        """
        texts = [msg.text for msg in messages]

        embeddings = self.model.encode(texts)

        for msg, emb in zip(messages, embeddings):
            msg.embedding = emb

        return embeddings
    
    def encode_query(self, text):
        return self.model.encode(text)


# =============================
# Vector Index
# =============================


class VectorIndex:
    """
    Stores message embeddings and performs similarity search.
    """

    def __init__(self):
        self.messages = []
        self.embeddings = []

    def add(self, message):
        """
        Add message and its embedding into index.

        """
        if message.embedding is None:
            raise ValueError("Message must have embedding before indexing.")

        self.messages.append(message)
        self.embeddings.append(message.embedding)

    def remove(self, message):
        """
        Remove message from index.

        """
        if message in self.messages:
            idx = self.messages.index(message)
            self.messages.pop(idx)
            self.embeddings.pop(idx)

    def search(self, query_embedding, top_k=5):
        """
        Return top_k most similar messages.

        """

        if not self.embeddings:
            return []

        matrix = np.vstack(self.embeddings)

        similarities = cosine_similarity(
            [query_embedding],
            matrix
        )[0]

        indices = similarities.argsort()[::-1][:top_k]

        return [self.messages[i] for i in indices]
    
# =============================
# Eviction Strategy
# =============================

class EvictionStrategy:
    """
    Adaptive pruning based on importance + recency

    """

    def linear_recency_score(self, message, max_age=3600):

        age = time.time() - message.timestamp
        score = 1 - (age / max_age)

        return max(0.0, score)

    def choose_eviction(self, messages):

        if not messages:
            return None

        scored = []

        for msg in messages:

            recency = self.linear_recency_score(msg)

            usage_score = min(msg.access_count / 5, 1.0)  # normalize usage

            survival_score = (
                0.5 * msg.importance +
                0.3 * recency +
                0.2 * usage_score
            )


            scored.append((survival_score, msg))

        scored.sort(key=lambda x: x[0])

        # lowest survival score gets removed
        return scored[0][1]
    
# =============================
# Memory Store
# =============================

class MemoryStore:

    def __init__(
        self,
        max_size,
        vectorizer,
        vector_index,
        eviction_strategy,
        importance_calculator
    ):

        self.messages = []
        self.max_size = max_size

        self.vectorizer = vectorizer
        self.vector_index = vector_index
        self.eviction_strategy = eviction_strategy
        self.importance_calculator = importance_calculator

    def add_message(self, message):

        # Auto-calculate importance 
        if message.importance == 0.0:
            message.importance = self.importance_calculator.calculate(
                message.text,
                message.role
            )

        # Generate embedding (state-aware vectorizer)
        self.vectorizer.encode(message)

        # Adaptive eviction
        if len(self.messages) >= self.max_size:

            to_remove = self.eviction_strategy.choose_eviction(self.messages)

            if to_remove:
                self.messages.remove(to_remove)
                self.vector_index.remove(to_remove)

        # Store message
        self.messages.append(message)

        # Real-time index update
        self.vector_index.add(message)



# =============================
# Retriever
# =============================

class Retriever:
    """
    Combines semantic similarity, importance, and recency
    to retrieve the most relevant messages.
    """

    def __init__(self, vectorizer, vector_index):
        self.vectorizer = vectorizer
        self.vector_index = vector_index

    def linear_recency_score(self, message, max_age=3600):
        """
        Linear decay based on message age.
        """
        age = time.time() - message.timestamp
        score = 1 - (age / max_age)
        return max(0.0, score)

    def retrieve(self, query, top_k=5):

        # Step 1: encode query
        query_embedding = self.vectorizer.model.encode(query)

        # Step 2: semantic candidates
        candidate_pool = max(top_k * 4, 10)

        candidates = self.vector_index.search(query_embedding, candidate_pool)


        ranked = []

        for msg in candidates:

            # semantic similarity
            semantic = cosine_similarity(
                [query_embedding],
                [msg.embedding]
            )[0][0]

            # importance
            importance = msg.importance

            # recency
            recency = self.linear_recency_score(msg)

            # weighted final score
            final_score = (
                0.7 * semantic +
                0.2 * importance +
                0.1 * recency
            )

            ranked.append((final_score, msg))

        ranked.sort(key=lambda x: x[0], reverse=True)

        results = [msg for _, msg in ranked[:top_k]]

        # increase usage count
        for msg in results:
            msg.access_count += 1

        return results

    

