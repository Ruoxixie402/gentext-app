from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel

app = FastAPI(
    title="Bigram Text Generation API",
    description="Generate text using a Bigram language model",
    version="1.0.0"
)


corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]




bigram_model = BigramModel(corpus)


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
