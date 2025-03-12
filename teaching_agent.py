import os
from typing import List, Dict, Any, Tuple
import base64
from pathlib import Path

# PDF processing libraries
import PyPDF2
import fitz  # PyMuPDF

# LangGraph for reflection architecture
from langgraph.graph import MessageGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Import our custom Gemini LLM wrapper
from gemini_llm import GenaiLLM


class TeachingAgent:
    """
    AI Teaching Agent specialized for programming education (Python, SQL, etc.).
    Processes lecture PDFs and creates coding-focused learning materials with reflection.
    Uses a reflection mechanism to improve the quality of generated content and ensure
    programming best practices are followed.
    """

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.2, google_api_key=None):
        """Initialize the teaching agent with Google Gemini LLM."""
        if not google_api_key:
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError(
                    "Google API key must be provided either directly or via GOOGLE_API_KEY environment variable")

        # Set the API key in the environment
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Initialize our custom Gemini LLM wrapper
        self.llm = GenaiLLM(
            model_name=model_name,
            temperature=temperature
        )

        # Create the reflection graph
        self.graph = self._build_reflection_graph()

    def _build_reflection_graph(self) -> MessageGraph:
        """Build the reflection graph using LangGraph."""
        builder = MessageGraph()

        # Add nodes for generate and reflect
        builder.add_node("generate", self._generation_node)
        builder.add_node("reflect", self._reflection_node)

        # Set the entry point
        builder.set_entry_point("generate")

        # Add conditional edges
        builder.add_conditional_edges("generate", self._should_continue)
        builder.add_edge("reflect", "generate")

        return builder.compile()

    def _should_continue(self, state: List[BaseMessage]):
        """Determine if we should continue the reflection loop."""
        # Limit to 2 reflection cycles (4 messages: 2 generations + 2 reflections)
        # Reduced from 3 cycles to minimize API calls
        if len(state) > 4:
            return END
        return "reflect"

    def _generation_node(self, state: List[BaseMessage]) -> List[BaseMessage]:
        """Generate content based on the current state."""
        # If this is the first generation, start with the initial request
        if len(state) == 1:
            # Extract the task from the first message
            task = state[0].content

            # Generate the initial response with programming education focus
            try:
                response = self.llm.invoke([
                    SystemMessage(content="""You are an expert programming educator specializing in teaching 
                    languages like Python, SQL, JavaScript, and other programming concepts. 
                    Your goal is to create clear, executable code examples and practical explanations that follow
                    programming best practices. Focus on helping learners understand both the syntax and the underlying
                    concepts. Include practical, real-world applications wherever possible."""),
                    HumanMessage(content=task)
                ])
            except Exception as e:
                print(f"Error in generation node: {e}")
                # Provide a fallback response
                response = AIMessage(
                    content="I couldn't generate a proper response. The input might be too short or unclear.")
        else:
            # Use reflection feedback for improved generation
            messages = state.copy()
            try:
                response = self.llm.invoke(messages)
            except Exception as e:
                print(f"Error in subsequent generation: {e}")
                # Provide a fallback response
                response = AIMessage(content="I couldn't generate an improved response based on the reflection.")

        # Return updated state with the new generation appended
        return state + [response]

    def _reflection_node(self, state: List[BaseMessage]) -> List[BaseMessage]:
        """Reflect on and critique the latest generation."""
        # Get the latest generation
        latest_generation = state[-1].content

        # Create a reflection prompt
        reflection_prompt = f"""
        Review the following generated content as an expert educator:

        {latest_generation}

        Please critique this content considering:
        1. Educational effectiveness - Is it clear, accurate, and pedagogically sound?
        2. Completeness - Does it cover all necessary aspects?
        3. Engagement - Will it engage and motivate learners?

        Provide specific, constructive feedback on how to improve this content.
        """

        # Generate programming-focused reflection
        try:
            reflection_response = self.llm.invoke([
                SystemMessage(content="""You are an experienced software developer and programming educator
                reviewing educational content. Provide thoughtful, constructive criticism focusing on:
                1. Code quality and correctness - Is the code following best practices, is it efficient, and would it run as expected?
                2. Educational value - Does it clearly explain programming concepts and build appropriate mental models?
                3. Practical application - Does it connect theory to real-world programming scenarios?
                4. Progressional learning - Does it scaffold concepts appropriately for beginners while challenging advanced learners?"""),
                HumanMessage(content=reflection_prompt)
            ])
        except Exception as e:
            print(f"Error in reflection node: {e}")
            # Provide a fallback reflection
            reflection_response = AIMessage(
                content="The content looks good, but consider adding more examples and improving the explanations.")

        # Return updated state with reflection appended
        return state + [reflection_response]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        text = ""

        # Check if file exists first
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: '{pdf_path}'")

        print(f"Extracting text from: {pdf_path}")

        # Try PyMuPDF first for better text extraction
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")

            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text += page.extract_text() or ""
            except Exception as e2:
                print(f"PyPDF2 extraction failed: {e2}")
                raise Exception(f"Failed to extract text from PDF '{pdf_path}'. Error: {str(e2)}")

        # Ensure we have actual text content
        text = text.strip()
        if not text:
            raise Exception(f"No text content extracted from PDF: '{pdf_path}'")

        print(f"Successfully extracted {len(text)} characters from PDF")

        return text

    def summarize_content(self, text: str) -> str:
        """Generate a summary of the lecture content."""
        # Ensure text is not empty
        if not text or len(text.strip()) < 10:
            raise ValueError("Insufficient text content to summarize")

        # Limit text to avoid token limits, but ensure it's not empty
        text_sample = text[:25000]  # Reduced from 25000 to avoid token limits
        if len(text_sample.strip()) < 10:
            raise ValueError("Text sample is too short after truncation")

        # Initialize the reflection process with a summarization task
        initial_prompt = f"""
        Create a concise but comprehensive summary of the following programming lecture content:

        {text_sample}

        The summary should:
        1. Highlight key programming concepts and important points
        2. Identify main topics covered
        3. Be organized in a logical structure
        4. Include any notable coding examples or techniques
        """

        # Print prompt length for debugging
        print(f"Summarization prompt length: {len(initial_prompt)} characters")

        # Run the reflection graph
        messages = [HumanMessage(content=initial_prompt)]
        try:
            result = self.graph.invoke(messages)
            # Extract the final summary from the result
            summary = result[-1].content
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            # Provide a fallback response rather than failing completely
            return "This lecture appears to contain programming content. The material covers various programming topics and examples."

    def detect_programming_language(self, text: str) -> str:
        """
        Detect the main programming language discussed in the lecture content.

        Args:
            text (str): The lecture content text

        Returns:
            str: Detected programming language (e.g., "python", "sql", "javascript")
        """
        # Take just the first part of the text to save tokens
        text_sample = text[:1500]

        # Create a simple prompt to detect the programming language
        detect_prompt = f"""
        Based on the following lecture content, determine the main programming language being discussed.
        Return only the language name in lowercase (e.g., "python", "sql", "javascript", "java", "c++", etc.).
        If no specific programming language is discussed, respond with "general".

        Content excerpt:
        {text_sample}
        """

        try:
            # Use a single LLM call (no reflection needed for this simple task)
            response = self.llm.invoke([
                SystemMessage(content="You analyze text and extract the main programming language discussed."),
                HumanMessage(content=detect_prompt)
            ])

            # Extract and clean the language name
            language = response.content.strip().lower()

            # Handle cases where no specific language is detected
            if "not" in language or "multiple" in language or "cannot" in language or len(language) > 20:
                return "general"

            return language
        except Exception as e:
            print(f"Error detecting language: {e}")
            return "general"  # Default to general if there's an error

    def create_with_reflection(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generic function to create content with reflection.
        Uses the reflection graph for any type of content.

        Args:
            prompt (str): The prompt for content generation
            system_prompt (str, optional): System prompt for initial message

        Returns:
            str: The generated content after reflection
        """
        # Create initial message
        if system_prompt:
            # When we have a system prompt, use it with the reflection process
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            initial_response = self.llm.invoke(messages)
            messages = [HumanMessage(content=prompt), initial_response]
        else:
            # No system prompt, just use the human message directly
            messages = [HumanMessage(content=prompt)]

        # Run the reflection graph with the messages
        try:
            result = self.graph.invoke(messages)
            # Extract the final content from the result
            content = result[-1].content
            return content
        except Exception as e:
            print(f"Error in reflection process: {e}")
            # If reflection fails, try a direct generation
            try:
                if system_prompt:
                    response = self.llm.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt)
                    ])
                else:
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content
            except Exception as e2:
                print(f"Error in fallback generation: {e2}")
                return f"Unable to generate content. Error: {str(e2)}"

    def create_code_examples(self, text: str, language: str) -> str:
        """Generate code examples for the detected programming language with reflection."""
        # Take a sample of the text to avoid token limits
        text_sample = text[:3000]

        # Create a prompt for code examples
        code_prompt = f"""
        Create 2-3 clear, educational code examples in {language.upper()} based on the concepts in this lecture:

        {text_sample}

        For each example:
        1. Focus on a key concept from the lecture
        2. Include comments explaining how the code works
        3. Show expected output where appropriate
        4. Keep examples simple but practical
        """

        # System prompt for code examples
        system_prompt = f"""You are an expert {language} programmer creating educational code examples.
        Your examples should be clear, concise, and follow best practices.
        Format your code with proper syntax highlighting and clear comments.
        Use simple examples that demonstrate important concepts."""

        # Use reflection to generate and improve the code examples
        return self.create_with_reflection(code_prompt, system_prompt)

    def create_practice_exercises(self, text: str, language: str) -> str:
        """
        Generate hands-on practice exercises with solutions using reflection.

        Args:
            text (str): The lecture content
            language (str): The programming language

        Returns:
            str: Formatted practice exercises with solutions
        """
        # Take a sample of the text to avoid token limits
        text_sample = text[:3000]

        # Create a prompt for practice exercises
        practice_prompt = f"""
        Create 3 practical coding exercises in {language.upper()} based on the concepts in this lecture:

        {text_sample}

        For each exercise:
        1. Create a clear problem statement
        2. Provide starter code when appropriate
        3. Include a complete solution
        4. Explain how the solution works

        Format each exercise as:

        ## Exercise X: [Title]

        ### Problem
        [Clear description of what to implement]

        ### Starter Code
        ```{language}
        [Starter code if appropriate]
        ```

        ### Solution
        ```{language}
        [Complete solution code]
        ```

        ### Explanation
        [Detailed explanation of the solution]
        """

        # System prompt for practice exercises
        system_prompt = f"""You are an expert {language} programming teacher creating educational exercises.
        Your exercises should be challenging but achievable, building on concepts from the lecture.
        Each exercise should include a clear problem statement, starter code, complete solution, and explanation.
        Make sure the exercises progress in difficulty and cover different aspects of the lecture content."""

        # Use reflection to generate and improve the practice exercises
        return self.create_with_reflection(practice_prompt, system_prompt)

    def create_assessment(self, text: str, language: str) -> Dict[str, Any]:
        """Generate programming-focused assessment questions and answers with reflection."""
        # Take a sample of the text to avoid token limits
        text_sample = text[:3000]

        # Create a prompt for assessment questions
        assessment_prompt = f"""
        Create 5 assessment questions with answers about {language} programming based on this lecture:

        {text_sample}

        Include a mix of:
        - Conceptual questions that test understanding of key concepts
        - Code understanding questions where students analyze code snippets
        - Practical application questions that ask how to solve specific problems

        Make questions specific to the lecture content, not generic programming questions.

        Format as:

        Question 1: [Question text]
        Answer 1: [Answer text]

        Question 2: [Question text] 
        Answer 2: [Answer text]

        And so on.
        """

        # System prompt for assessment questions
        system_prompt = f"""You are creating assessment questions for {language} programming students.
        Your questions should test understanding of concepts, not just memorization.
        Include different question types and difficulties, focusing on the key concepts from the lecture.
        Provide detailed answers that explain not just what the answer is, but why it's correct."""

        # Use reflection to generate and improve the assessment questions
        qa_text = self.create_with_reflection(assessment_prompt, system_prompt)

        # Parse the response into questions and answers
        questions_answers = self._parse_qa(qa_text)
        return questions_answers

    def _parse_qa(self, qa_text: str) -> Dict[str, Any]:
        """Parse questions and answers text into a structured format."""
        lines = qa_text.strip().split('\n')
        questions = []
        answers = []

        current_q = ""
        current_a = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.lower().startswith("question"):
                # Save previous Q&A if we're starting a new question
                if current_q and current_a:
                    questions.append(current_q)
                    answers.append(current_a)

                # Extract the new question
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_q = parts[1].strip()
                else:
                    current_q = ""
                current_a = ""

            elif line.lower().startswith("answer"):
                # Extract the answer
                parts = line.split(':', 1)
                if len(parts) > 1:
                    current_a = parts[1].strip()
                else:
                    current_a = ""

        # Add the last Q&A
        if current_q and current_a:
            questions.append(current_q)
            answers.append(current_a)

        # If parsing failed, provide fallback
        if not questions or not answers:
            questions = ["What is this lecture about?"]
            answers = ["This lecture covers programming concepts and techniques."]

        return {
            "questions": questions,
            "answers": answers
        }

    def process_lecture(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a programming lecture PDF and return comprehensive teaching materials.

        Args:
            pdf_path (str): Path to the lecture PDF file

        Returns:
            Dict with language detection, summary, code examples, practice exercises, and assessment materials
        """
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            print(f"Successfully extracted {len(text)} characters from PDF")

            # Detect the programming language
            language = self.detect_programming_language(text)
            print(f"Detected language: {language}")

            # Generate summary
            summary = self.summarize_content(text)
            print("Generated summary")

            # Generate code examples
            code_examples = self.create_code_examples(text, language)
            print("Generated code examples")

            # Generate practice exercises
            practice_exercises = self.create_practice_exercises(text, language)
            print("Generated practice exercises")

            # Generate assessment questions
            assessment = self.create_assessment(text, language)
            print("Generated assessment questions")

            return {
                "language": language,
                "summary": summary,
                "code_examples": code_examples,
                "practice_exercises": practice_exercises,
                "assessment": assessment
            }
        except Exception as e:
            print(f"Error in process_lecture: {e}")
            # Provide a minimal result if processing fails
            return {
                "language": "general",
                "summary": "Unable to process the lecture content fully.",
                "code_examples": "Code examples could not be generated.",
                "practice_exercises": "Practice exercises could not be generated.",
                "assessment": {
                    "questions": ["What topics does this lecture cover?"],
                    "answers": ["The lecture appears to cover programming topics."]
                }
            }


# Test code if run directly
if __name__ == "__main__":
    import sys

    try:
        # Initialize the agent first
        agent = TeachingAgent()

        if len(sys.argv) > 1:
            # Use the file path provided as command line argument
            pdf_path = sys.argv[1]
            print(f"Processing PDF from command line: {pdf_path}")
        else:
            # Use a default file in the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pdf_path = os.path.join(current_dir, "lecture.pdf")
            print(f"Processing default PDF: {pdf_path}")

        # Test just the text extraction first
        text = agent.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters from PDF")
        print("First 200 characters:")
        print(text[:200])

        # Only try processing if extraction worked
        if len(text) > 100:  # Arbitrary minimum length
            print("\nProcessing full PDF...")
            result = agent.process_lecture(pdf_path)
            print("\nSummary:")
            print(result["summary"])
            print("\nDetected Language:", result["language"])
    except Exception as e:
        print(f"Error: {e}")