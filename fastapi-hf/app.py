from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

app = FastAPI(title="Text2Text API")

MODEL_NAME = "google/flan-t5-small"

# Load model and tokenizer once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


class RequestBody(BaseModel):
    prompt: str
    max_length: int = 100
    num_return_sequences: int = 1


@app.get("/")
async def root():
    return {"message": "Welcome to the Text2Text API"}


@app.post("/generate")
async def generate(body: RequestBody):
    # Use the pipeline to generate text
    outputs = pipe(
        body.prompt,
        max_length=body.max_length,
        num_return_sequences=body.num_return_sequences
    )
    # Normalize response
    results = [
        o.get("generated_text") or o.get("summary_text") or str(o)
        for o in outputs
    ]
    return {"generated_text": results}
