import os
from typing import List, Dict, Any, Tuple
import PyPDF2
import fitz  # PyMuPDF
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangGraph for reflection architecture
from langgraph.graph import MessageGraph, END
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

# Import our custom Gemini LLM wrapper
from gemini_llm import GenaiLLM


class ReflexionTeachingAgent:
    """
    AI Teaching Agent specialized for programming education using Reflexion framework.
    Incorporates ReAct (Reasoning + Acting) and Chain-of-Thought (CoT) techniques
    to enhance educational content generation with structured reasoning.
    """

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.2, google_api_key=None, dataset_path: str | None = None):
        """Initialize the teaching agent with Google Gemini LLM and load dataset."""
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

        # Create the reflexion graph
        self.graph = self._build_reflexion_graph()

        # Load the local instruction dataset for RAG
        if dataset_path is None:
            dataset_path = os.environ.get(
                "INSTRUCTION_DATASET_PATH",
                os.path.join(os.path.dirname(__file__), "data", "instructions.csv"),
            )

        self.dataset_connected = False
        self.dataset = None
        self.vectorizer = None
        self.matrix = None
        if dataset_path and os.path.exists(dataset_path):
            try:
                data = pd.read_csv(dataset_path)
                if {'instruction', 'input', 'output'} <= set(data.columns):
                    docs = (data['instruction'].fillna('') + ' ' + data['input'].fillna('')).tolist()
                    self.vectorizer = TfidfVectorizer(max_features=5000)
                    self.matrix = self.vectorizer.fit_transform(docs)
                    self.dataset = data
                    self.dataset_connected = True
                    print(f"Loaded instruction dataset from {dataset_path} with {len(data)} rows")
                else:
                    print("Dataset missing required columns; ignoring")
            except Exception as e:
                print(f"Failed to load dataset: {e}")

    def _build_reflexion_graph(self) -> MessageGraph:
        """Build the reflexion graph using LangGraph."""
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
        """Determine if we should continue the reflexion loop."""
        # Limit to 2 reflection cycles (4 messages: 2 generations + 2 reflections)
        if len(state) > 4:
            return END
        return "reflect"

    def _generation_node(self, state: List[BaseMessage]) -> List[BaseMessage]:
        """Generate content based on the current state using ReAct or CoT approaches."""
        # If this is the first generation, start with the initial request
        if len(state) == 1:
            # Extract the task from the first message
            task = state[0].content

            # Generate the initial response with programming education focus using ReAct prompting
            try:
                response = self.llm.invoke([
                    SystemMessage(content="""You are an expert programming educator specializing in teaching 
                    languages like Python, SQL, JavaScript, and other programming concepts. 

                    Follow the ReAct approach:
                    1. First THINK by analyzing the problem and breaking it down into steps
                    2. Then ACT by creating clear code examples following best practices

                    IMPORTANT FORMATTING INSTRUCTIONS:
                    - Put "Thought:" sections OUTSIDE of code blocks
                    - Put "Action:" sections OUTSIDE of code blocks
                    - Always place code examples in proper markdown code blocks with language specified
                    - Never mix "Thought:" or "Action:" markers inside code blocks

                    For example:

                    Thought: I'll demonstrate list operations in Python.

                    Action: Here's a code example for list creation and indexing:

                    ```python
                    # Create a list
                    my_list = [10, 20, 30, 40, 50]

                    # Access elements
                    print(my_list[0])  # Output: 10
                    ```

                    Thought: Now I'll explain list slicing.

                    Action: Here's a code example for list slicing:

                    ```python
                    # List slicing
                    my_list = [10, 20, 30, 40, 50]
                    print(my_list[1:4])  # Output: [20, 30, 40]
                    ```

                    Include Chain-of-Thought reasoning to make your explanations step-by-step and clear.
                    Focus on helping learners understand both the syntax and the underlying concepts.
                    Include practical, real-world applications wherever possible."""),
                    HumanMessage(content=task)
                ])
            except Exception as e:
                print(f"Error in generation node: {e}")
                # Provide a fallback response
                response = AIMessage(
                    content="I couldn't generate a proper response. The input might be too short or unclear.")
        else:
            # Use reflexion feedback for improved generation
            messages = state.copy()
            try:
                response = self.llm.invoke(messages)
            except Exception as e:
                print(f"Error in subsequent generation: {e}")
                # Provide a fallback response
                response = AIMessage(content="I couldn't generate an improved response based on the reflexion.")

        # Return updated state with the new generation appended
        return state + [response]

    def _reflection_node(self, state: List[BaseMessage]) -> List[BaseMessage]:
        """Reflect on and critique the latest generation using structured evaluation."""
        # Get the latest generation
        latest_generation = state[-1].content

        # Create a structured reflection prompt using CoT
        reflection_prompt = f"""
        Review the following generated educational content:

        {latest_generation}

        Follow a Chain-of-Thought approach to evaluate this content systematically:

        Step 1: Analyze the pedagogical structure and approach
        Step 2: Evaluate the technical accuracy and code quality
        Step 3: Assess the clarity of explanations and examples
        Step 4: Consider whether it addresses different learning styles
        Step 5: Provide specific suggestions for improvement

        For each step, provide your reasoning and specific feedback.
        """

        # Generate programming-focused reflection
        try:
            reflection_response = self.llm.invoke([
                SystemMessage(content="""You are an experienced software developer and programming educator
                reviewing educational content. Use a Chain-of-Thought approach to provide structured, 
                step-by-step feedback on:

                1. Code quality and correctness - Is the code following best practices, is it efficient, and would it run as expected?
                2. Educational value - Does it clearly explain programming concepts and build appropriate mental models?
                3. Practical application - Does it connect theory to real-world programming scenarios?
                4. Progressional learning - Does it scaffold concepts appropriately for beginners while challenging advanced learners?

                Structure your feedback with clear reasoning at each step."""),
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
        """Generate a summary of the lecture content using Chain-of-Thought reasoning."""
        # Ensure text is not empty
        if not text or len(text.strip()) < 10:
            raise ValueError("Insufficient text content to summarize")

        # Limit text to avoid token limits, but ensure it's not empty
        text_sample = text[:25000]
        if len(text_sample.strip()) < 10:
            raise ValueError("Text sample is too short after truncation")

        # Initialize the reflexion process with a CoT summarization task
        initial_prompt = f"""
        Create a concise but comprehensive summary of the following programming lecture content.

        Use Chain-of-Thought reasoning to:
        1. First identify the main programming topics and concepts
        2. Then analyze how these concepts relate to each other
        3. Finally synthesize this into a structured summary

        Lecture content:
        {text_sample}

        Your summary should:
        1. Highlight key programming concepts and important points
        2. Identify main topics covered
        3. Be organized in a logical structure
        4. Include any notable coding examples or techniques
        """

        # Print prompt length for debugging
        print(f"Summarization prompt length: {len(initial_prompt)} characters")

        # Run the reflexion graph
        messages = [HumanMessage(content=initial_prompt)]
        try:
            result = self.graph.invoke(messages)
            # Extract the final summary from the result
            summary = result[-1].content
            return summary
        except Exception as e:
            print(f"Error during summarization: {e}")
            # Provide a fallback response
            return "This lecture appears to contain programming content. The material covers various programming topics and examples."

    def detect_programming_language(self, text: str) -> str:
        """
        Detect the main programming language discussed in the lecture content with improved accuracy.
        """
        # Take a larger sample of the text to improve detection
        text_sample = text[:3000]

        # Create a structured analytic prompt for language detection
        detect_prompt = f"""
        Analyze the following programming lecture content and determine the main programming language being discussed.

        Content excerpt:
        {text_sample}

        Follow these steps:
        1. Look for explicit language names mentioned (Python, JavaScript, Java, etc.)
        2. Identify language-specific syntax (semicolons, brackets, indentation patterns)
        3. Detect characteristic keywords and functions
        4. Note any code examples and their syntax patterns

        Common language indicators:
        - Python: def, import, print(), indentation, no semicolons, range(), list comprehensions
        - JavaScript: var/let/const, function, semicolons, braces for blocks, console.log()
        - Java: public class, static void main, System.out.println(), strong typing
        - C++: #include, cout, cin, int main(), namespaces
        - SQL: SELECT, FROM, WHERE, JOIN, database terminology

        Return ONLY ONE of these language names in lowercase:
        python, javascript, java, cpp, csharp, sql, ruby, go, php, swift, typescript, kotlin, rust, html, css, general

        If you cannot confidently determine a specific language or the content covers multiple languages equally, return "general".

        RETURN ONLY THE LANGUAGE NAME, NO OTHER TEXT OR EXPLANATION.
        """

        try:
            # Use a direct approach to avoid citation issues
            response = self.llm.invoke([
                SystemMessage(
                    content="You are a programming language analyzer. Your only task is to identify the primary programming language in educational content. Return only a single word - the language name."),
                HumanMessage(content=detect_prompt)
            ])

            # Clean and normalize the response
            language = response.content.strip().lower()

            # Check for validity and normalize
            valid_languages = [
                "python", "javascript", "java", "cpp", "csharp", "sql",
                "ruby", "go", "php", "swift", "typescript", "kotlin",
                "rust", "html", "css", "general"
            ]

            # Remove any extra text (explanations, etc.)
            if "\n" in language:
                language = language.split("\n")[0].strip()

            # Further normalize by extracting just the language name
            for valid_lang in valid_languages:
                if valid_lang in language:
                    return valid_lang

            # If we can't match to a known language, default to general
            print(f"Language detection returned '{language}' which was normalized to 'general'")
            return "general"

        except Exception as e:
            print(f"Error in language detection: {e}")
            return "general"  # Default to general if there's an error

    def create_with_reflexion(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generic function to create content with reflexion.
        Uses the reflexion graph for any type of content.

        Args:
            prompt (str): The prompt for content generation
            system_prompt (str, optional): System prompt for initial message

        Returns:
            str: The generated content after reflexion
        """
        # Create initial message
        if system_prompt:
            # When we have a system prompt, use it with the reflexion process
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ]
            initial_response = self.llm.invoke(messages)
            messages = [HumanMessage(content=prompt), initial_response]
        else:
            # No system prompt, just use the human message directly
            messages = [HumanMessage(content=prompt)]

        # Run the reflexion graph with the messages
        try:
            result = self.graph.invoke(messages)
            # Extract the final content from the result
            content = result[-1].content
            return content
        except Exception as e:
            print(f"Error in reflexion process: {e}")
            # If reflexion fails, try a direct generation
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
        """Generate code examples using ReAct approach."""
        # Take a sample of the text to avoid token limits
        text_sample = text[:3000]

        # Create a ReAct prompt for code examples
        code_prompt = f"""
        Create 2-3 clear, educational code examples in {language.upper()} based on the concepts in this lecture.

        Use the ReAct approach:

        Thought: First, identify key concepts from the lecture that would benefit from code examples
        Action: Create a code example that teaches this concept

        IMPORTANT FORMATTING INSTRUCTIONS:
        - Keep each "Thought:" section SEPARATE from code blocks
        - Keep each "Action:" section SEPARATE from code blocks 
        - Place all code in proper markdown code blocks with triple backticks
        - Use clear section headers for each example
        - Add meaningful comments within the code blocks
        - NEVER include "Thought:" or "Action:" markers inside code blocks

        For each example:
        1. Include a "Thought:" section explaining what concept you're demonstrating
        2. Include an "Action:" section introducing the example
        3. Provide a code block with the example, including comments
        4. Show expected output where appropriate

        Lecture excerpt:
        {text_sample}
        """

        # System prompt for code examples with CoT and ReAct
        system_prompt = f"""You are an expert {language} programmer creating educational code examples.
        Use both Chain-of-Thought and ReAct approaches to create clear, step-by-step explanations.
        For Chain-of-Thought, break down your reasoning into explicit steps.
        For ReAct, alternate between thinking about the educational goal and acting by creating code.

        CRITICAL FORMATTING RULES:
        1. ALWAYS place "Thought:" and "Action:" markers OUTSIDE of code blocks
        2. NEVER include these markers inside ```{language} code blocks
        3. Place ALL code examples inside proper markdown code blocks with language specified
        4. Structure each example with clear Thought/Action separation
        5. Use proper markdown formatting for all content

        Your examples should be:
        - Clear and concise
        - Following best practices
        - Well-commented
        - Demonstrating important concepts from the lecture
        """

        # Use reflexion to generate and improve the code examples
        return self.create_with_reflexion(code_prompt, system_prompt)

    def create_practice_exercises(self, text: str, language: str) -> str:
        """
        Generate hands-on practice exercises with solutions using ReAct and CoT.
        """
        # Take a sample of the text to avoid token limits
        text_sample = text[:3000]

        # Create a prompt for practice exercises using ReAct and CoT
        practice_prompt = f"""
        Create 3 practical coding exercises in {language.upper()} based on the concepts in this lecture.

        Use the ReAct (Reasoning + Acting) approach:

        Thought: First, identify key learning objectives from the lecture
        Action: Design an exercise that tests understanding of these objectives
        Thought: Consider what starter code would be helpful
        Action: Create starter code that guides without giving away the solution
        Thought: Develop a complete solution with step-by-step reasoning
        Action: Provide a detailed explanation of the solution

        Lecture excerpt:
        {text_sample}

        For each exercise:
        1. Create a clear problem statement
        2. Provide starter code when appropriate
        3. Include a complete solution
        4. Explain how the solution works using Chain-of-Thought reasoning

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
        [Detailed explanation of the solution with step-by-step reasoning]
        """

        # System prompt for practice exercises
        system_prompt = f"""You are an expert {language} programming teacher creating educational exercises.
        Use Chain-of-Thought reasoning to break down complex problems into understandable steps.
        Your exercises should be challenging but achievable, building on concepts from the lecture.
        Each exercise should include a clear problem statement, starter code, complete solution, and explanation.
        Make sure the exercises progress in difficulty and cover different aspects of the lecture content."""

        # Use reflexion to generate and improve the practice exercises
        return self.create_with_reflexion(practice_prompt, system_prompt)

    def create_assessment(self, text: str, language: str) -> Dict[str, Any]:
        """Generate 10 assessment questions using RAG and CoT."""
        text_sample = text[:3000]

        # Try to get reference examples from dataset for RAG
        examples_text = ""
        selected_examples: List[Dict[str, str]] = []
        dataset_connected = self.dataset_connected
        if self.dataset_connected:
            try:
                examples = self._get_similar_examples(text_sample, num_examples=5)
                print("Selected dataset examples for assessment generation:")
                for idx, ex in enumerate(examples, 1):
                    print(f"Example {idx}: {ex['instruction'][:80]}")
                    examples_text += (
                        f"Instruction: {ex['instruction']}\n"
                        f"Input: {ex['input']}\n"
                        f"Output: {ex['output']}\n\n"
                    )
                selected_examples = examples
            except Exception as e:
                print(f"Example retrieval failed: {e}")

        assessment_prompt = f"""
        Create 10 assessment questions with answers about {language} programming based on this lecture.
        At least 8 questions should require the student to write or analyze code.

        Use the following reference examples to guide style and expected answers:
        {examples_text}

        Use Chain-of-Thought reasoning to:
        1. Identify key concepts that should be assessed
        2. Formulate questions testing these concepts
        3. Provide detailed explanations for the answers

        Lecture excerpt:
        {text_sample}

        Format as:

        Question 1: [Question text]
        Answer 1: [Answer text with step-by-step explanation]

        Question 2: [Question text]
        Answer 2: [Answer text with step-by-step explanation]

        And so on up to Question 10.
        """

        system_prompt = f"""You are creating assessment questions for {language} programming students.
        Your questions must mix conceptual understanding with practical coding tasks.
        Provide clear, step-by-step explanations in the answers."""

        qa_text = self.create_with_reflexion(assessment_prompt, system_prompt)
        questions_answers = self._parse_qa(qa_text)
        questions_answers["dataset_examples"] = selected_examples
        questions_answers["dataset_connected"] = dataset_connected
        return questions_answers

    def _parse_qa(self, qa_text: str) -> Dict[str, Any]:
        """Parse questions and answers text into a structured format."""
        import re

        pattern = re.compile(
            r"Question\s*\d+\s*:(.*?)\n\s*Answer\s*\d+\s*:(.*?)(?=Question\s*\d+\s*:|$)",
            re.DOTALL | re.IGNORECASE,
        )
        matches = pattern.findall(qa_text)

        questions = []
        answers = []
        for q, a in matches:
            q = q.strip()
            a = a.strip()
            if q and a:
                questions.append(q)
                answers.append(a)

        # Fallback if regex failed
        if not questions or not answers:
            questions = ["What is this lecture about?"]
            answers = ["This lecture covers programming concepts and techniques."]

        # Ensure equal length and at most 10 items
        length = min(len(questions), len(answers), 10)
        return {
            "questions": questions[:length],
            "answers": answers[:length],
        }

    def _get_similar_examples(self, text: str, num_examples: int = 5) -> List[Dict[str, str]]:
        """Retrieve similar instructions from the loaded dataset using TF-IDF."""
        if self.dataset is None or self.vectorizer is None or self.matrix is None:
            return []

        query_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = sims.argsort()[-num_examples:][::-1]

        examples = []
        for idx in top_idx:
            row = self.dataset.iloc[idx]
            examples.append({
                'instruction': str(row.get('instruction', '')),
                'input': str(row.get('input', '')),
                'output': str(row.get('output', ''))
            })

        return examples

    def process_lecture(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a programming lecture PDF and return comprehensive teaching materials
        with both process and final result versions.

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

            # Generate summary using Chain-of-Thought
            summary = self.summarize_content(text)
            print("Generated summary")

            # Generate code examples with ReAct (includes thought process)
            code_examples = self.create_code_examples(text, language)
            print("Generated code examples with thought process")

            # Generate practice exercises using ReAct and CoT
            practice_exercises = self.create_practice_exercises(text, language)
            print("Generated practice exercises")

            # Generate assessment questions using CoT
            assessment = self.create_assessment(text, language)
            print("Generated assessment questions")

            # Post-process to create clean versions
            code_examples_clean = self.clean_thought_process(code_examples)
            print("Created clean version of code examples")

            practice_exercises_clean = self.clean_thought_process(practice_exercises)
            print("Created clean version of practice exercises")

            return {
                "language": language,
                "summary": summary,
                "code_examples": code_examples,
                "code_examples_clean": code_examples_clean,
                "practice_exercises": practice_exercises,
                "practice_exercises_clean": practice_exercises_clean,
                "assessment": assessment,
                "dataset_connected": assessment.get("dataset_connected", False)
            }
        except Exception as e:
            print(f"Error in process_lecture: {e}")
            # Provide a minimal result if processing fails
            return {
                "language": "general",
                "summary": "Unable to process the lecture content fully.",
                "code_examples": "Code examples could not be generated.",
                "code_examples_clean": "Code examples could not be generated.",
                "practice_exercises": "Practice exercises could not be generated.",
                "practice_exercises_clean": "Practice exercises could not be generated.",
                "assessment": {
                    "questions": ["What topics does this lecture cover?"],
                    "answers": ["The lecture appears to cover programming topics."]
                },
                "dataset_connected": False
            }

    def clean_thought_process(self, content: str) -> str:
        """
        Post-process content to remove Thought/Action markers and create a clean version.

        Args:
            content: The original content with Thought/Action sections

        Returns:
            Clean version with thought process removed
        """
        if not content or len(content) < 10:
            return content

        try:
            # Create a prompt to clean the content
            clean_prompt = f"""
            Below is educational content with "Thought:" and "Action:" markers showing the reasoning process.
            Please create a clean version of this content by:

            1. Removing all "Thought:" sections completely
            2. Removing "Action:" markers but keeping the action content
            3. Preserving all code blocks and their comments exactly as they are
            4. Keeping all examples, explanations, and outputs intact
            5. Ensuring proper markdown formatting with headings and code blocks
            6. Removing any placeholders like "THOUGHT_PLACEHOLDER_X"

            Original content:
            ```
            {content}
            ```

            Output only the clean version without any explanation.
            """

            response = self.llm.invoke([
                SystemMessage(
                    content="You are a content formatter that removes thought process markers from educational content while preserving the core material."),
                HumanMessage(content=clean_prompt)
            ])

            clean_content = response.content.strip()

            # If API call fails or returns empty, fall back to regex-based cleaning
            if not clean_content or len(clean_content) < 20:
                return self._regex_clean_thought_process(content)

            return clean_content

        except Exception as e:
            print(f"Error in clean_thought_process: {e}")
            # Fall back to regex-based cleaning
            return self._regex_clean_thought_process(content)

    def _regex_clean_thought_process(self, content: str) -> str:
        """
        Use regex to clean thought process markers when the API approach fails.
        """
        import re

        # Skip if content is too short
        if not content or len(content) < 10:
            return content

        try:
            # Remove "Thought:" sections (patterns like "Thought: text here Action:" or "Thought: text here ```")
            cleaned = re.sub(r'Thought:.*?(?=Action:|```|\*\*Example|\*\*|##|$)', '', content, flags=re.DOTALL)

            # Remove "Action:" markers but keep the content
            cleaned = re.sub(r'Action:\s*', '', cleaned)

            # Remove any "THOUGHT_PLACEHOLDER_X" markers
            cleaned = re.sub(r'THOUGHT_PLACEHOLDER_\d+', '', cleaned)

            # Clean up excess whitespace and newlines
            cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)

            # If cleaning produced something too short, return original
            if len(cleaned.strip()) < len(content) * 0.3:
                return content

            return cleaned.strip()

        except Exception as e:
            print(f"Error in regex cleaning: {e}")
            return content


    def create_code_analysis_prompt(self, user_code: str, exercise_description: str, expected_solution: str | None = None) -> str:
        """Create a prompt for analyzing user code submissions."""
        solution_part = f"Reference Solution:\n```\n{expected_solution}\n```" if expected_solution else ""
        prompt = f"""
        Analyze the following code submission for a programming exercise and provide detailed feedback:

        Exercise Description:
        {exercise_description}

        User Submission:
        ```
        {user_code}
        ```

        {solution_part}

        Use your programming expertise to:
        1. Determine if the solution is correct
        2. Identify any bugs, errors, or inefficiencies
        3. Detect conceptual misunderstandings or misconceptions
        4. Provide specific, constructive feedback
        5. Suggest hints that guide the student without giving away the complete solution

        Additionally provide your own best attempt at the solution code.

        Return your analysis in JSON with these fields:
        - is_correct
        - misconceptions
        - suggestions
        - hints
        - score (0-100 reflecting how correct the solution is)
        - model_solution (string containing your suggested solution code)
        """
        return prompt

    def analyze_code(self, user_code: str, exercise_description: str, expected_solution: str | None = None) -> Dict[str, Any]:
        """Analyze code using the Reflexion framework."""
        try:
            prompt = self.create_code_analysis_prompt(user_code, exercise_description, expected_solution)
            analysis = self.create_with_reflexion(prompt)

            import json, re
            try:
                result = json.loads(analysis)
            except json.JSONDecodeError:
                json_match = re.search(r'```json\s*(.*?)\s*```', analysis, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    result = {
                        "is_correct": "correct" in analysis.lower() and "incorrect" not in analysis.lower(),
                        "misconceptions": [],
                        "suggestions": [],
                        "hints": [],
                        "score": 0.0,
                    }

            if "score" not in result:
                result["score"] = 100.0 if result.get("is_correct") else 0.0
            else:
                try:
                    result["score"] = float(result["score"])
                except Exception:
                    result["score"] = 100.0 if result.get("is_correct") else 0.0

            if "model_solution" not in result:
                result["model_solution"] = ""
            return result
        except Exception as e:
            print(f"Error in code analysis: {e}")
            return {
                "is_correct": False,
                "misconceptions": ["Unable to fully analyze your code"],
                "suggestions": ["Please review your code for syntax errors"],
                "hints": ["Check the exercise requirements carefully"],
                "score": 0.0,
                "model_solution": "",
            }

# Test code if run directly
if __name__ == "__main__":
    import sys

    try:
        # Initialize the agent first
        agent = ReflexionTeachingAgent()

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
