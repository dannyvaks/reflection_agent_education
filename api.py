from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from tempfile import NamedTemporaryFile
import shutil
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file (for Google API key)
load_dotenv()

# Import our ReflexionTeachingAgent (renamed from TeachingAgent)
from reflexion_teaching_agent import ReflexionTeachingAgent

# Check if Google API key is set
if not os.environ.get("GOOGLE_API_KEY"):
    print("WARNING: GOOGLE_API_KEY environment variable is not set.")
    print("Make sure to set it before initializing the ReflexionTeachingAgent.")

app = FastAPI(title="AI Teaching Assistant API",
              description="API for processing programming lecture PDFs using the Reflexion framework with ReAct and Chain-of-Thought techniques")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development (adjust in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve our HTML frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize our teaching agent
# Will use the GOOGLE_API_KEY environment variable
try:
    agent = ReflexionTeachingAgent()
except ValueError as e:
    print(f"Error initializing ReflexionTeachingAgent: {e}")
    print("The API will start, but endpoints requiring the agent will fail.")
    agent = None


@app.post("/process-lecture/", response_model=Dict[str, Any])
async def process_lecture(file: UploadFile = File(...), language: str = None):
    """
    Process a programming lecture PDF and return comprehensive teaching materials.

    The processing uses the Reflexion framework with ReAct and Chain-of-Thought techniques
    to generate high-quality educational content.

    Args:
        file: Uploaded PDF file
        language: Optional programming language to focus on (if not specified, the agent will try to detect it)

    Returns:
        JSON with language detection, summary, code examples, practice exercises, and assessment questions/answers
    """
    # Check if agent is initialized
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="ReflexionTeachingAgent not initialized. Check if GOOGLE_API_KEY is properly set."
        )

    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save the uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        # Process the lecture
        result = agent.process_lecture(temp_path)

        # If a specific language was requested but differs from detected, include a note
        if language and language.lower() != result["language"]:
            result[
                "note"] = f"You requested content for {language}, but the lecture appears to be about {result['language']}. The materials have been generated accordingly."

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)


@app.get("/health/")
async def health_check():
    """Health check endpoint to verify the API and agent are functioning."""
    return {"status": "healthy", "agent_initialized": agent is not None}


@app.get("/")
async def redirect_to_frontend():
    """Redirect root path to the frontend."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


if __name__ == "__main__":
    # Start the FastAPI server
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)