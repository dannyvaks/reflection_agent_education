from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from tempfile import NamedTemporaryFile
import shutil
from typing import Dict, Any, List
from pydantic import BaseModel

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


class CodeAnalysisRequest(BaseModel):
    code: str
    exercise_id: str # Could be a description of the exercise
    expected_solution: str | None = None

# Pydantic models for the new grading endpoint
class StudentAnswerItem(BaseModel):
    question: str
    student_answer: str
    correct_answer: str

class GradeAssessmentRequest(BaseModel):
    answers: List[StudentAnswerItem]

class GradedAnswerItem(BaseModel):
    question: str
    student_answer: str
    correct_answer: str
    score: float
    feedback: str

class GradeAssessmentResponse(BaseModel):
    graded_answers: List[GradedAnswerItem]
    total_score: float


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

@app.post("/analyze-code/")
async def analyze_code_endpoint(submission: CodeAnalysisRequest):
    """Analyze code with the teaching agent and provide feedback."""
    if agent is None:
        raise HTTPException(status_code=500, detail="ReflexionTeachingAgent not initialized")
    feedback = agent.analyze_code(submission.code, submission.exercise_id, submission.expected_solution)
    return feedback

@app.post("/grade-assessment/", response_model=GradeAssessmentResponse)
async def grade_assessment(request: GradeAssessmentRequest):
    """
    Grades a list of student answers against correct answers using the ReflexionTeachingAgent.
    """
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="ReflexionTeachingAgent not initialized. Check if GOOGLE_API_KEY is properly set."
        )

    graded_answers_list = []
    total_score = 0.0

    for item in request.answers:
        try:
            grading_result = agent.grade_student_answer(
                question=item.question,
                student_answer=item.student_answer,
                correct_answer=item.correct_answer
            )

            current_score = grading_result.get("score", 0.0)
            feedback = grading_result.get("feedback", "Error in grading.")

            graded_answers_list.append(
                GradedAnswerItem(
                    question=item.question,
                    student_answer=item.student_answer,
                    correct_answer=item.correct_answer,
                    score=current_score,
                    feedback=feedback
                )
            )
            total_score += current_score
        except Exception as e:
            # Handle potential errors during grading of a single item
            # Log the error and add a placeholder for the failed item
            print(f"Error grading item: {item.question} - {str(e)}")
            graded_answers_list.append(
                GradedAnswerItem(
                    question=item.question,
                    student_answer=item.student_answer,
                    correct_answer=item.correct_answer,
                    score=0.0,
                    feedback=f"Failed to grade this answer: {str(e)}"
                )
            )
            # Optionally, decide if you want to add 0 to total_score or handle differently

    # Ensure total_score is capped (though individual scores are 0-10, summing 10 Qs = 100 max)
    # This is more of a safeguard if number of questions varies or scoring logic changes.
    # For 10 questions, max is 100. If there are N questions, max is N*10.
    # For now, let's assume a general cap if needed, or rely on individual caps.
    # total_score = min(total_score, 100.0) # Example cap if assessment is always out of 100

    return GradeAssessmentResponse(graded_answers=graded_answers_list, total_score=total_score)


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