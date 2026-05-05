from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from rank_bm25 import BM25Okapi


EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
COLLECTION = "bonus_memory"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


@dataclass
class ProfileStore:
    """Tiny feature-store-like table for the bonus POC."""

    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, user_id: str) -> dict[str, Any]:
        return self.profiles.setdefault(
            user_id,
            {
                "preferred_language": "vi",
                "reading_speed_wpm": 220,
                "topic_affinity": "cloud",
                "queries_last_hour": 0,
                "recent_topics": [],
            },
        )

    def record_query(self, user_id: str, query: str) -> None:
        profile = self.get(user_id)
        profile["queries_last_hour"] += 1
        for topic in ("kubernetes", "cloud", "security", "autoscaling", "ai"):
            if topic in query.lower() and topic not in profile["recent_topics"]:
                profile["recent_topics"].insert(0, topic)
        profile["recent_topics"] = profile["recent_topics"][:5]


class HybridMemoryAgent:
    def __init__(self) -> None:
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        self.profile_store = ProfileStore()
        self._ids = itertools.count()
        self._memories: list[dict[str, Any]] = []

    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        chunks = self._chunk(text)
        vectors = list(self.embedder.embed(chunks))
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = next(self._ids)
            payload = {"user_id": user_id, "text": chunk}
            self._memories.append({"id": point_id, **payload})
            points.append(PointStruct(id=point_id, vector=vector.tolist(), payload=payload))
        self.client.upsert(collection_name=COLLECTION, points=points)

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + user profile features and return context."""
        profile = self.profile_store.get(user_id)
        self.profile_store.record_query(user_id, query)
        semantic = self._semantic_search(query, user_id, top_k=5)
        keyword = self._keyword_search(query, user_id, top_k=5)
        fused = self._rrf(keyword, semantic)[:3]

        memories = "\n".join(f"- {m['text']}" for m in fused) or "- No memory yet."
        recent_topics = ", ".join(profile["recent_topics"]) or "none"
        return (
            f"User profile: language={profile['preferred_language']}, "
            f"reading_speed={profile['reading_speed_wpm']}wpm, "
            f"topic_affinity={profile['topic_affinity']}.\n"
            f"Recent activity: queries_last_hour={profile['queries_last_hour']}, "
            f"recent_topics={recent_topics}.\n"
            f"Top memories:\n{memories}"
        )

    @staticmethod
    def _chunk(text: str, max_chars: int = 600) -> list[str]:
        parts = [p.strip() for p in re.split(r"(?<=[.!?。])\s+", text) if p.strip()]
        chunks: list[str] = []
        current = ""
        for part in parts or [text.strip()]:
            if len(current) + len(part) + 1 <= max_chars:
                current = f"{current} {part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks

    def _semantic_search(self, query: str, user_id: str, top_k: int) -> list[dict[str, Any]]:
        q_vec = next(self.embedder.embed([query])).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
            limit=top_k,
        )
        return [{"id": p.id, "text": p.payload["text"]} for p in result.points]

    def _keyword_search(self, query: str, user_id: str, top_k: int) -> list[dict[str, Any]]:
        memories = [m for m in self._memories if m["user_id"] == user_id]
        if not memories:
            return []
        bm25 = BM25Okapi([_tokens(m["text"]) for m in memories])
        scores = bm25.get_scores(_tokens(query))
        ranked = sorted(range(len(memories)), key=lambda i: -scores[i])[:top_k]
        return [{"id": memories[i]["id"], "text": memories[i]["text"]} for i in ranked]

    @staticmethod
    def _rrf(keyword: list[dict[str, Any]], semantic: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
        scores: dict[int, float] = {}
        meta: dict[int, dict[str, Any]] = {}
        for hits in (keyword, semantic):
            for rank, item in enumerate(hits, start=1):
                scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (k + rank)
                meta.setdefault(item["id"], item)
        return [meta[item_id] for item_id, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
