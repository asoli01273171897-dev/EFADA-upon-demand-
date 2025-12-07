import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from inngest.experimental import ai
from data_loader import _load_and_chunk_pdf, embed_texts
from vectordp import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertresult, RAGChunkAndSRC

# Load environment variables
load_dotenv()

# Initialize Inngest client
inngest_client = inngest.Inngest(
    app_id="my-first-rag-app",
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# 🚀 Ingest PDF function
@inngest_client.create_function(
    fn_id="RAG: Ingest pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    print("🚀 Inngest function triggered")

    def _load(ctx: inngest.Context) -> RAGChunkAndSRC:
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", pdf_path)
        print(f"📄 Loading PDF from: {pdf_path}")
        try:
            chunks = _load_and_chunk_pdf(pdf_path)
            print(f"✅ Loaded {len(chunks)} chunks from PDF")
            return RAGChunkAndSRC(chunks=chunks, source_id=source_id)
        except Exception as e:
            print(f"❌ Error while loading PDF: {e}")
            raise

    def _upsert(chunks_and_src: RAGChunkAndSRC) -> RAGUpsertresult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        print(f"🧩 Preparing to embed {len(chunks)} chunks for source: {source_id}")
        try:
            vecs = embed_texts(chunks)
            print("✅ Embeddings created")
            ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
            payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
            QdrantStorage().upsert(ids, vecs, payloads)
            print("✅ Upsert into Qdrant complete")
            return RAGUpsertresult(ingested=len(chunks))
        except Exception as e:
            print(f"❌ Error during embedding/upsert: {e}")
            raise

    try:
        chunks_and_src = await ctx.step.run(
            "load_and_chunk",
            lambda: _load(ctx),
            output_type=RAGChunkAndSRC
        )
        ingested = await ctx.step.run(
            "embed_and_upsert",
            lambda: _upsert(chunks_and_src),
            output_type=RAGUpsertresult
        )
        print(f"🎉 Ingestion finished: {ingested.ingested} chunks stored")
        return ingested.model_dump()
    except Exception as e:
        print("❌ Error in rag_ingest_pdf:", e)
        raise

# 🔍 Query PDF function with step-based embedding and search
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    print("🔍 RAG query function triggered")
    try:
        question = ctx.event.data["question"]
        top_k = int(ctx.event.data.get("top_k", 5))
        print(f"❓ Question: {question} | Top K: {top_k}")

        def search(q: str, k: int) -> RAGSearchResult:
            query_vec = embed_texts([q])[0]
            store = QdrantStorage()
            found = store.search(query_vec, k)
            return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

        result = await ctx.step.run(
            step_id="embed-and-search",
            fn=lambda: search(question, top_k),
            output_type=RAGSearchResult
        )

        print(f"✅ Found {len(result.contexts)} results")
        return result.model_dump()
    except Exception as e:
        print(f"❌ Error in rag_query_pdf_ai: {e}")
        raise

# 🌐 FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello RAG!"}

# ✅ Register both functions with Inngest
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
