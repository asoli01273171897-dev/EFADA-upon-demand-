from pydantic import BaseModel
from typing import List

class RAGChunkAndSRC(BaseModel):
    chunks: List[str]
    source_id: str

class RAGUpsertresult(BaseModel):
    ingested: int

class RAGSearchResult(BaseModel):
    contexts: List[str]
    sources: List[str]

class RAGQueryResult(BaseModel):
    answer: str
    contexts: List[str]
    sources: List[str]
