import os
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding model details
EMBED_MODEL = "text-embedding-3-small"   # cheaper model, 1536 dimensions
EMBED_DIM = 1536


def _load_and_chunk_pdf(file_path: str):
    """Load a PDF using PyPDF2 and split into text chunks."""
    print(f"🔍 Attempting to read PDF: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ PDF not found at {file_path}")

    try:
        reader = PdfReader(file_path)
        texts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                texts.append(text)
            else:
                print(f"⚠️ Page {i} has no extractable text")

        if not texts:
            raise ValueError("❌ No text could be extracted from PDF — it may be scanned or image-only")

        print(f"✅ Extracted text from {len(texts)} pages")

        # Simple chunking: split into ~1000-character chunks with 200 overlap
        chunks = []
        for t in texts:
            while len(t) > 1000:
                chunks.append(t[:1000])
                t = t[800:]  # overlap of 200
            if t:
                chunks.append(t)

        print(f"✅ Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"❌ PDF loading failed: {e}")
        raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    print(f"🔑 Embedding {len(texts)} texts with OpenAI")
    embeddings = []
    try:
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            response = client.embeddings.create(
                input=batch,
                model=EMBED_MODEL,
            )
            embeddings.extend([item.embedding for item in response.data])
        print("✅ Embeddings created with OpenAI")
        return embeddings
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        raise
