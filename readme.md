# AI Teaching Agent with Reflexion

This project implements an AI teaching assistant that processes lecture PDFs, generates educational content, and creates assessment questions using the Reflexion framework, which combines ReAct and Chain-of-Thought techniques in a multi-turn improvement cycle.

## Features

- PDF processing for lecture materials
- Programming language detection
- AI-powered summarization with Reflexion capabilities
- Generation of code examples following best practices
- Creation of practice exercises with solutions
- Generation of assessment questions and answers
- 10-question assessments with dataset-based examples (at least 8 with code tasks)
- Selected dataset examples are shown in Learning Process view for transparency
- Web interface with dual view mode (Learning Process and Final Result)
- API for programmatic integration
- Automatic code feedback using the Reflexion agent with per-question scores
  (each question contributes up to 10% toward the overall grade)
- Friendly Run Code button analyzes your solution and displays the model's code
  afterward without actually executing your submission
- Friendly visual feedback for code submissions with an overall score summary
- Model solution shown after you submit your answer so you can compare
- Dataset connection status displayed, with retrieved examples listed when available
- Single "Run Code" button shows the analysis results and grade

## Methodology and Architecture

### Reflexion Framework and Architecture

This project implements a Reflexion architecture as described in the paper "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023). Our implementation follows a specific pattern:

1. **Generation Node with ReAct**: The initial content is generated using the ReAct (Reasoning + Acting) approach, where the model explicitly shows its thinking process and then takes actions based on that reasoning.

2. **Reflection Node with CoT**: The generated content is then critically evaluated using Chain-of-Thought (CoT) reasoning, which provides structured feedback for improvements.

3. **Feedback Loop**: The reflection is fed back into the generation node, creating an iterative improvement cycle that enhances the quality of the educational content.

This cyclical process continues for multiple iterations, producing increasingly refined content that benefits from both the ReAct approach's transparency and the structured analysis of CoT reasoning.

### ReAct Approach (Reasoning + Acting)

The Generation Node uses the ReAct pattern to structure its thinking:

1. **Reasoning (Thought)**: The agent explicitly shows its reasoning process before taking action, labeled with "Thought:" sections that reveal the step-by-step logic.

2. **Acting (Action)**: After reasoning, the agent takes concrete actions (like generating code or designing exercises), labeled with "Action:" sections.

This pattern makes the teaching assistant's thought process transparent and more educational, showing students not just what to do but how to think about programming problems.

### Chain-of-Thought (CoT) Reasoning

The Reflection Node employs Chain-of-Thought reasoning to evaluate and improve the content:

1. **Step-by-Step Evaluation**: The content is analyzed systematically across multiple dimensions.
2. **Structured Feedback**: Specific, actionable improvement suggestions are provided.
3. **Progressive Refinement**: Each iteration builds upon previous reflections.

This systematic evaluation approach ensures that educational materials improve through multiple reflection cycles.

## Dual View Interface

The web interface provides two viewing modes:

1. **Learning Process View**: Shows the complete thought process using the ReAct framework, including all "Thought:" and "Action:" sections. This view is ideal for educational purposes, allowing students to understand the reasoning behind the code and explanations.

2. **Final Result View**: Presents a clean, distilled version of the content with the thought process removed. This view is better for professional use or when focusing only on the implementation details without the underlying reasoning.

Users can toggle between these views at any time using the buttons at the top of the results section.

## Why This Architecture?

This Reflexion architecture combining ReAct and CoT was chosen for several reasons:

1. **Comprehensive Improvement Cycle**: The generation-reflection-improvement loop creates increasingly refined content.

2. **Transparent Educational Process**: ReAct's explicit reasoning and action steps help students understand not just what to do but why to do it.

3. **Structured Evaluation**: Chain-of-Thought reflection provides systematic analysis that identifies specific areas for improvement.

4. **Multi-Turn Refinement**: Content benefits from multiple iterations of feedback and enhancement.

5. **Flexibility**: The dual view allows users to either see the full reasoning process or focus on the polished final result.

## RAG-Based Assessment Generation

Assessment questions are created using Retrieval Augmented Generation (RAG).
During lecture processing, the agent retrieves up to five similar examples from
a Kaggle dataset of Python instructions. Each example provides an `instruction`,
`input`, and `output` field:

```
instruction: Write a function that adds two numbers.
input: 2 3
output: 5
```

These examples are included in the prompt so the language model can mimic their
style and expected outputs while still using the Reflexion loop to refine the
questions. The final output lists ten questions in the form:

```
Question 1: ...
Answer 1: ...
```

The response also indicates whether the dataset was successfully loaded so the
frontend can show a connection status.

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Google API key with access to Gemini models

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/reflexion-teaching-agent.git
   cd reflexion-teaching-agent
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
   # Path to Kaggle code instruction dataset (train.csv)
   INSTRUCTION_DATASET_PATH=/path/to/train.csv
   ```

### Running the Application

1. Start the FastAPI server:
   ```
   uvicorn api:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Upload a PDF lecture file and view the generated summary, code examples, practice exercises, and assessment questions.

4. Use the toggle buttons at the top of the results to switch between "Learning Process" and "Final Result" views.

## API Documentation

After starting the server, you can access the API documentation at:
```
http://localhost:8000/docs
```

Key endpoints:
```
POST /process-lecture/  - Process a lecture PDF
POST /analyze-code/     - Analyze code submissions and provide feedback
```

## Project Structure

- `reflexion_teaching_agent.py`: Core AI agent implementing the Reflexion framework with ReAct and CoT
- `api.py`: FastAPI application for the web API
- `gemini_llm.py`: Wrapper for the Google Gemini model
- `static/index.html`: Frontend interface with dual view capability
- `requirements.txt`: Project dependencies

## Key Methods

### reflexion_teaching_agent.py

- `_build_reflexion_graph()`: Creates the graph structure for the reflection process
- `_generation_node()`: Generates educational content using ReAct approach
- `_reflection_node()`: Critically evaluates generated content using CoT reasoning
- `_should_continue()`: Determines if additional reflection cycles should occur
- `create_code_examples()`: Uses the Reflexion graph to generate educational code examples
- `create_practice_exercises()`: Uses the Reflexion graph to create programming exercises
- `create_assessment()`: Uses the Reflexion graph to design assessment questions
- `_get_similar_examples()`: Retrieves dataset examples for RAG question generation
- `clean_thought_process()`: Post-processes content to create clean versions without thought process
- `process_lecture()`: Orchestrates the full lecture processing workflow

### api.py

- `process_lecture()`: Handles PDF uploads and interfaces with the ReflexionTeachingAgent
- `health_check()`: API endpoint for monitoring system status

## Using Google Gemini

This project uses Google's Gemini LLM models. Make sure your API key has access to Gemini models for best results. The implementation includes specific handling for citation and recitation issues that can occur with Gemini models.

## References

- Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366.
- Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
- Wei, J., et al. (2022). Chain of Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.