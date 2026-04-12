from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    chunk: DocumentChunk
    similarity_score: float

@dataclass
class QueryIntent:
    intent_type: str  # e.g., 'troubleshooting', 'informational', 'comparison'
    confidence: float
