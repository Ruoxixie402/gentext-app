from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel
from contextlib import asynccontextmanager
import spacy
from fastapi import Query, Request, File, UploadFile
import os
from pathlib import Path
from app.image_classifier import DEFAULT_CHECKPOINT, MAX_IMAGE_BYTES, ImageClassifier


@asynccontextmanager
async def lifespan(app):
    app.state.nlp = spacy.load("en_core_web_md")
    checkpoint = Path(os.getenv("CNN_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    app.state.image_classifier = ImageClassifier(checkpoint) if checkpoint.is_file() else None
    yield

app = FastAPI(
    title="Text Generation, Word Embedding, and Image Classification API",
    description="Generate bigram text, query spaCy word vectors, and classify images with a CIFAR-10 CNN",
    version="1.2.0",
    lifespan=lifespan,
)


corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]




bigram_model = BigramModel(corpus)


@app.post("/classify")
def classify_image(request: Request, image: UploadFile = File(...)):
    """Classify a PNG/JPEG upload into one of the ten CIFAR-10 classes."""
    data = image.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 5 MiB limit.")
    classifier = request.app.state.image_classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Trained classifier checkpoint is unavailable.")
    try:
        return classifier.predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/embedding")
def word_embedding(request: Request, word: str = Query(..., min_length=1)):
    """Return the pretrained spaCy vector for one English word."""
    word = word.strip()
    tokens = request.app.state.nlp.make_doc(word)
    if len(tokens) != 1 or tokens[0].is_space:
        raise HTTPException(status_code=400, detail="Provide exactly one non-empty word.")
    token = tokens[0]
    if not token.has_vector:
        raise HTTPException(status_code=404, detail="No pretrained vector is available for this word.")
    vector = token.vector.tolist()
    return {"word": word, "model": "en_core_web_md", "dimension": len(vector), "embedding": vector}


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():

    return {
        "message": "Bigram Text Generation API is running"
    }



@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }



@app.post("/generate")
def generate_text(request: TextGenerationRequest):

    # Check whether length is valid
    if request.length <= 0:
        raise HTTPException(
            status_code=400,
            detail="Length must be greater than 0."
        )

    
    if not request.start_word.strip():
        raise HTTPException(
            status_code=400,
            detail="start_word cannot be empty."
        )

    try:

        
        generated_text = bigram_model.generate_text(
            request.start_word,
            request.length
        )

        
        return {
            "start_word": request.start_word,
            "length": request.length,
            "generated_text": generated_text
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
