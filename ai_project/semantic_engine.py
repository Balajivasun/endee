import hashlib
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from endee_client import EndeeClient
from data_models import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class SemanticIngestionEngine:
    def __init__(self, index_name: str = "enterprise_kb", similarity_threshold: float = 0.96):
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise

        self.client = EndeeClient()
        self._ensure_index()

    def _ensure_index(self):
        self.client.create_index(self.index_name, self.dimension)

    def _generate_chunk_id(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        if not words:
            return chunks
            
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += (chunk_size - overlap)
        return chunks

    def process_document(self, document_text: str, source_name: str) -> Dict[str, int]:
        chunks = self._chunk_text(document_text)
        stats = {"total": len(chunks), "ingested": 0, "deduplicated": 0}

        if not chunks:
            return stats

        embeddings = self.model.encode(chunks).tolist()

        for chunk_text, vector in zip(chunks, embeddings):
            existing = self.client.search_vectors(self.index_name, vector, top_k=1)
            
            is_duplicate = False
            if existing:
                top_match = existing[0]
                if top_match.get('score', 0) >= self.similarity_threshold:
                    is_duplicate = True
                    
            if is_duplicate:
                stats["deduplicated"] += 1
            else:
                chunk_id = self._generate_chunk_id(chunk_text)
                metadata = {"chunk_id": chunk_id, "source": source_name, "text": chunk_text}
                self.client.insert_vectors(self.index_name, [vector], [metadata])
                stats["ingested"] += 1

        return stats

    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_vector = self.model.encode([query])[0].tolist()
        raw_results = self.client.search_vectors(self.index_name, query_vector, top_k=top_k)
        
        parsed_results = []
        for item in raw_results:
            meta = item.get('metadata', {})
            chunk = DocumentChunk(
                chunk_id=meta.get('chunk_id', ''),
                text=meta.get('text', ''),
                source=meta.get('source', 'Unknown'),
                metadata=meta
            )
            parsed_results.append(SearchResult(chunk=chunk, similarity_score=item.get('score', 0.0)))
            
        return parsed_results
