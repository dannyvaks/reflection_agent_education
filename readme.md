# AI Teaching Agent with Reflection

This project implements an AI teaching assistant that processes lecture PDFs, generates summaries, and creates assessment questions using a reflection-based AI architecture.

## Features

- PDF processing for lecture materials
- AI-powered summarization with reflection capabilities
- Generation of assessment questions and answers
- Web interface for uploading PDFs and viewing results
- API for programmatic integration

## How It Works

The system uses a reflection architecture built with LangGraph to improve the quality of generated content:

1. The "generate" node creates initial content (System 1 thinking)
2. The "reflect" node critiques the content (System 2 thinking)
3. This process loops to refine and improve the output

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Google API key with access to Gemini models

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/teaching-agent.git
   cd teaching-agent
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Google API key:
   ```
   export GOOGLE_API_KEY=your_api_key_here  # On Windows: set GOOGLE_API_KEY=your_api_key_here
   ```
   
   Alternatively, create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

### Running the Application

1. Start the FastAPI server:
   ```
   uvicorn app:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Upload a PDF lecture file and view the generated summary and assessment questions.

## API Documentation

After starting the server, you can access the API documentation at:
```
http://localhost:8000/docs
```

## Project Structure

- `teaching_agent.py`: Core AI agent with reflection capabilities
- `app.py`: FastAPI application for the web API
- `static/index.html`: Frontend interface
- `requirements.txt`: Project dependencies

## Using Google Gemini

This project uses Google's Gemini LLM models. Make sure your API key has access to these models.