# FastAPI + Hugging Face + Docker (CPU)


This example shows how to deploy a Hugging Face text2text model behind a FastAPI app and package it with Docker.


## From project root (where Dockerfile is)
docker build -t fastapi-hf:latest .


## Run


docker run --rm -p 8000:8000 fastapi-hf:latest


Then open: http://localhost:8000/docs for interactive API docs.


## Notes
- The Dockerfile installs a CPU-only torch wheel from the official PyTorch CPU index to avoid the "executable stack" and similar issues.
- If you have an NVIDIA GPU and want GPU support, replace the torch installation with suitable CUDA wheels and use a base image that supports CUDA. Also note container must be run with `--gpus` and appropriate drivers.
- For production-scale usage consider:
- Serving with `gunicorn` + `uvicorn.workers.UvicornWorker` or `uvicorn` behind a reverse proxy.
- Using model caching, batching, or a model server (e.g., TorchServe, BentoML, or Ray Serve).
- Storing model artifacts locally or mounting a volume with the model to avoid re-downloads.# Build (local)

 **Core Features**
- ✅ **RESTful API** with FastAPI for high-performance text generation
**Google's FLAN-T5-small model** for instruction-following text generation
**Docker containerization** for consistent deployment
 **Automatic API documentation** (Swagger UI & ReDoc)
**Input validation** using Pydantic models
 **Async/await support** for concurrent request handling

