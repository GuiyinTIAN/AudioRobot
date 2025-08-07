"""
Conversation Summarization Module

This module provides capabilities for summarizing conversations using different 
backends (OpenAI API, DeepSeek API, or offline local models).
"""

# Standard library imports
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
import openai
from transformers.pipelines import pipeline

# Import configuration
from config import (
    SUMMARIZER_TYPE,
    OPENAI_API_KEY,
    DEEPSEEK_API_KEY,
    OFFLINE_MODEL_NAME,
    ONLINE_MODEL_NAME
)


class BaseSummarizer(ABC):
    """Base class for conversation summarizers."""
    
    @abstractmethod
    def summarize(self, conversation_data) -> str:
        """
        Generate a summary for the given conversation data.
        
        Args:
            conversation_data: Conversation data to summarize (file path or data structure)
            
        Returns:
            str: Generated summary text
        """
        pass
    
    def prepare_conversation(self, conversation_data):
        """
        Prepare conversation data for summarization.
        
        Args:
            conversation_data: Conversation data (file path or data structure)
            
        Returns:
            str: Formatted conversation text
        """
        if isinstance(conversation_data, str) and os.path.exists(conversation_data):
            with open(conversation_data, 'r', encoding='utf-8') as f:
                data = json.load(f)
                conversation = data.get("conversation", [])
        else:
            conversation = conversation_data
        
        # Build conversation text with safer access methods
        conversation_lines = []
        for item in conversation:
            if isinstance(item, dict):
                speaker = item.get('speaker', 'Unknown')
                text = item.get('text', '')
                if text and text.strip():
                    conversation_lines.append(f"{speaker}: {text}")
        
        # More efficient than string concatenation
        conversation_text = "\n".join(conversation_lines)
        
        # Only print in debug mode
        if os.environ.get("DEBUG") == "1":
            print(f"[Debug] Conversation content:\n{conversation_text}\n")
        
        return conversation_text
    
    # def get_prompt(self, conversation_text):
    #     """
    #     Generate a prompt for summarization.
        
    #     Args:
    #         conversation_text (str): Prepared conversation text
            
    #     Returns:
    #         str: Formatted prompt for the summarizer
    #     """
    #     prompt = """
    #     Assume you are a doctor, please summarize the following conversation, noting:
    #     1. Ignore speech recognition errors and convert to correct medical terminology based on context
    #     2. Convert possible incorrect terms to standard medical terminology
        
    #     Conversation:
    #     {conversation_text}
    #     """.format(conversation_text=conversation_text)
        
    #     return prompt.strip()

    def get_prompt(self, conversation_text):
        """
        Generate a prompt for summarization.
        
        Args:
            conversation_text (str): Prepared conversation text
            
        Returns:
            str: Formatted prompt for the summarizer
        """
        prompt = """
        假设您是一名医生，请总结以下对话，注意：
        1. 忽略语音识别错误，并根据上下文将其转换为正确的医学术语
        2. 将可能不准确的术语转换为标准医学术语
        
        对话内容：
        {conversation_text}
        """.format(conversation_text=conversation_text)
        
        return prompt.strip()
    
    def summarize_to_file(self, conversation_data, output_dir, timestamp=None):
        """
        Generate a summary and save it to a file.
        
        Args:
            conversation_data: Conversation data to summarize
            output_dir (str): Directory to save the summary
            timestamp (str, optional): Timestamp for the filename
            
        Returns:
            str: Path to the saved summary file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        summary = self.summarize(conversation_data)
        
        # Handle None summary
        if summary is None:
            summary = "No summary could be generated."
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary to file
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("Conversation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary)
            f.write("\n\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Summary type: {self.__class__.__name__}\n")
        
        return summary_file


class OpenAISummarizer(BaseSummarizer):
    """Summarizer using OpenAI API."""
    
    def __init__(self):
        """Initialize the OpenAI API summarizer."""
        
        # Set API key
        api_key = OPENAI_API_KEY
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set in config.py or as environment variable.")
        
        openai.api_key = api_key
        self.model = ONLINE_MODEL_NAME or "gpt-3.5-turbo"
    
    def summarize(self, conversation_data):
        """
        Generate a summary using OpenAI API.
        
        Args:
            conversation_data: Conversation data to summarize
            
        Returns:
            str: Generated summary text
        """
        
        conversation_text = self.prepare_conversation(conversation_data)
        prompt = self.get_prompt(conversation_text)
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a conversation summarization expert. Extract key information and correct transcription errors based on context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            content = response.choices[0].message.content
            return content.strip() if content is not None else "No summary generated"
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return f"Summary generation failed: {str(e)}"


class DeepSeekSummarizer(BaseSummarizer):
    """Summarizer using DeepSeek API."""
    
    def __init__(self):
        """Initialize the DeepSeek API summarizer."""
        import requests
        self.requests = requests
        
        # Set API key
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        
        if not api_key:
            raise ValueError("DeepSeek API key not provided. Set in config.py or as environment variable.")
        
        self.api_key = api_key
        self.api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
        self.model = ONLINE_MODEL_NAME or "deepseek-chat"
    
    def summarize(self, conversation_data):
        """
        Generate a summary using DeepSeek API.
        
        Args:
            conversation_data: Conversation data to summarize
            
        Returns:
            str: Generated summary text
        """
        conversation_text = self.prepare_conversation(conversation_data)
        prompt = self.get_prompt(conversation_text)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a conversation summarization expert. Extract key information."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
        
        try:
            response = self.requests.post(
                f"{self.api_base}/chat/completions", 
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                raise Exception(f"API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"DeepSeek API call failed: {e}")
            return f"Summary generation failed: {str(e)}"


class OfflineSummarizer(BaseSummarizer):
    """Summarizer using offline models."""
    
    def __init__(self):
        """Initialize the offline summarizer."""
        try:     
            model_name = OFFLINE_MODEL_NAME or "facebook/bart-large-cnn"
            print(f"🔧 Loading local summarization model: {model_name}...")
            
            self.summarizer = pipeline("summarization", model=model_name)
        except ImportError:
            raise ImportError("transformers library not installed. Run: pip install transformers torch")
        except Exception as e:
            raise Exception(f"Failed to load offline model: {e}")
    
    def summarize(self, conversation_data):
        """
        Generate a summary using an offline model.
        
        Args:
            conversation_data: Conversation data to summarize
            
        Returns:
            str: Generated summary text
        """
        conversation_text = self.prepare_conversation(conversation_data)
        
        try:
            result = self.summarizer(conversation_text, 
                                     max_length=150, 
                                     min_length=30, 
                                     do_sample=False)
            return result[0]['summary_text']
        except Exception as e:
            print(f"Offline model summarization failed: {e}")
            return f"Summary generation failed: {str(e)}"


def choose_summarizer_type():
    """
    Let the user choose which summarization method to use.
    
    Returns:
        str: Selected summarizer type ('openai', 'deepseek', or 'offline')
    """
    print("\nChoose summarization method:")
    print("1. OpenAI API (requires API key)")
    print("2. DeepSeek API (requires API key)")
    print("3. Offline model (local processing, no API required)")
    print("4. Exit")
    
    while True:
        choice = input("Enter your choice (1/2/3/4): ").strip()
        if choice == '1':
            # Check API key
            api_key = os.environ.get("OPENAI_API_KEY", "") or OPENAI_API_KEY
            if not api_key:
                print("Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    continue
            return "openai"
        elif choice == '2':
            # Check API key
            api_key = os.environ.get("DEEPSEEK_API_KEY", "") or DEEPSEEK_API_KEY
            if not api_key:
                print("Warning: No DeepSeek API key found. Set DEEPSEEK_API_KEY environment variable.")
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    continue
            return "deepseek"
        elif choice == '3':
            try:
                # Check if transformers is installed
                import transformers
                return "offline"
            except ImportError:
                print("Warning: transformers library not installed. Offline model requires this dependency.")
                print("Run: pip install transformers torch")
                if input("Try to continue anyway? (y/n): ").lower() != 'y':
                    continue
                return "offline"
        elif choice == '4':
            print("Exiting summarization")
            exit(0)
        else:
            print("Invalid choice, please enter 1, 2, 3 or 4")




# =============================
# For testing purposes
# =============================
if __name__ == "__main__":
    import argparse
    import sys
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Conversation Summary Generator")
    parser.add_argument("-i", "--input", help="Path to input conversation JSON file")
    parser.add_argument("-o", "--output", help="Output directory (optional, defaults to input file directory)")
    args = parser.parse_args()
    
    # Show help info if no input file provided
    if not args.input and len(sys.argv) == 1:
        print("\n=== Conversation Summary Generator ===")
        print("Usage: python summarizer.py -i conversation_file.json [-o output_directory]")
        print("\nExamples:")
        print("  python summarizer.py -i outputs/session_20250722_175706/ai_conversation_20250722_175706.json")
        print("  python summarizer.py -i sample_conversation.json -o results\n")
        
        sys.exit(0)
    
    # Check if input file exists
    if args.input and not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    try:
        # Determine output directory
        if args.output:
            # User specified output directory
            output_dir = args.output
        else:
            # Default to input file's directory
            input_dir = os.path.dirname(os.path.abspath(args.input))
            output_dir = input_dir
            print(f"📁 Output directory: {output_dir}")

        # Let user choose summarization method
        summarizer_type = choose_summarizer_type()
        
        # Create summarizer based on selection
        if summarizer_type == "openai":
            summarizer = OpenAISummarizer()
            print("Selected: OpenAI API")
        elif summarizer_type == "deepseek":
            summarizer = DeepSeekSummarizer()
            print("Selected: DeepSeek API")
        else:
            summarizer = OfflineSummarizer()
            print("Selected: Offline model")
        
        print("🔄 Generating summary, please wait...")
        
        # Get input filename for metadata
        input_filename = os.path.basename(args.input)
        
        # Generate timestamp for file naming
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"🕒 Using timestamp: {current_timestamp}")
        
        # Generate summary
        summary = summarizer.summarize(args.input)
        
        # Create output file path
        summary_filename = f"summary_{current_timestamp}.txt"
        summary_file = os.path.join(output_dir, summary_filename)
        
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary file
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("Conversation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary)
            f.write("\n\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Summary type: {summarizer.__class__.__name__}\n")
            f.write(f"Source file: {input_filename}\n")
        
        print(f"✅ Summary saved to: {summary_file}")
        
        # Show summary preview
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_content = f.read()
            summary_parts = summary_content.split("=" * 50)
            if len(summary_parts) >= 3:
                summary_body = summary_parts[2].strip()
                preview = summary_body[:500]
                if len(summary_body) > 500:
                    preview += "..."
                print("\n=== Summary Preview ===\n")
                print(preview)
                print("\n" + "=" * 30)
            else:
                print("\nUnable to extract summary content for preview")
        
    except Exception as e:
        print(f"❌ Failed to generate summary: {e}")
        import traceback
        traceback.print_exc()