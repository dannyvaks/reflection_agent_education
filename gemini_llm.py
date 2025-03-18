import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
import google.generativeai as genai
from langgraph.graph import MessageGraph, END

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

genai.configure(api_key=GOOGLE_API_KEY)


class GenaiLLM:
    """
    A wrapper for the Google Generative AI model (Gemini)
    that provides a similar interface to the LangChain ChatOpenAI class.
    """

    def __init__(self, model_name="gemini-2.0-flash", temperature=0.2):
        """Initialize the Gemini model."""
        self.model_name = model_name
        self.temperature = temperature
        # Initialize the model with additional configuration to prevent citations
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": 0.95,
                "top_k": 40,
                # These settings help prevent recitation issues
                "response_mime_type": "text/plain",
            },
            # Configure safety settings
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        )

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Process a list of messages and return an AIMessage response.

        Args:
            messages (List[BaseMessage]): A list of LangChain message objects.

        Returns:
            AIMessage: The model's response as an AIMessage.
        """
        # Process system messages separately
        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

        # Combine system messages into a single prompt if present
        system_content = ""
        if system_messages:
            system_content = "\n\n".join([msg.content for msg in system_messages])
            system_content += "\n\nIMPORTANT: Do not use citations or quote sources directly. Present information as your own expertise."

        # Convert remaining messages to Gemini's format
        gemini_messages = []

        for i, message in enumerate(non_system_messages):
            if isinstance(message, HumanMessage):
                content = message.content
                # For the first human message, prepend system content if any
                if i == 0 and system_content:
                    content = f"{system_content}\n\n{content}"
                gemini_messages.append({
                    "role": "user",
                    "parts": [content]
                })
            elif isinstance(message, AIMessage):
                gemini_messages.append({
                    "role": "model",
                    "parts": [message.content]
                })

        # Handle the case where we might just have system messages
        if not gemini_messages and system_content:
            gemini_messages.append({
                "role": "user",
                "parts": [system_content]
            })

        # If there are no messages after processing, handle error
        if not gemini_messages:
            return AIMessage(content="No valid messages were provided.")

        # Try using the chat API first for multi-turn conversations
        if len(gemini_messages) > 1:
            try:
                chat = self.model.start_chat(history=[])

                # Add each message to the chat
                response = None
                for msg in gemini_messages:
                    if msg["role"] == "user":
                        response = chat.send_message(msg["parts"][0])

                # Return the last response
                if response and hasattr(response, 'text') and response.text:
                    return AIMessage(content=response.text)
                else:
                    raise Exception("No valid response from chat API")

            except Exception as e:
                print(f"Error in chat generation: {e}")
                # Fall through to the generate_content approach

        # For single messages or if chat API failed, use generate_content with
        # anti-citation instructions
        try:
            # Prepare the input with an anti-citation instruction
            if gemini_messages[-1]["role"] == "user":
                content = gemini_messages[-1]["parts"][0]
                content = f"{content}\n\nPlease provide an original response without citations or direct quotes."

                response = self.model.generate_content(content)
                if hasattr(response, 'text') and response.text:
                    return AIMessage(content=response.text)
                else:
                    raise Exception("Empty response from generate_content")
            else:
                # If the last message is not from user, use the last user message
                for msg in reversed(gemini_messages):
                    if msg["role"] == "user":
                        content = msg["parts"][0]
                        content = f"{content}\n\nPlease provide an original response without citations or direct quotes."

                        response = self.model.generate_content(content)
                        if hasattr(response, 'text') and response.text:
                            return AIMessage(content=response.text)
                        break

                # If we couldn't find a user message
                raise Exception("No valid user message found")

        except Exception as e:
            print(f"Error in generate_content: {e}")

            # Try one more approach with simplified content
            try:
                # Create a very simplified request
                simple_content = "Please respond to this programming education request with original content, no citations."
                if gemini_messages[-1]["role"] == "user":
                    simple_content += f" Request: {gemini_messages[-1]['parts'][0][:500]}"

                response = self.model.generate_content(simple_content)
                if hasattr(response, 'text') and response.text:
                    return AIMessage(content=response.text)
            except Exception as e2:
                print(f"Error in simplified generation: {e2}")

            # Final fallback
            return AIMessage(
                content="I apologize, but I was unable to process that request. Please try again with clearer instructions about the programming concepts you'd like me to explain.")