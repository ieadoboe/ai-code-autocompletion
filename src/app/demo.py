"""
Demo application for AI Code Completion using Gradio.
This provides a simple web interface for the code completion model.
"""
import os
import sys
import gradio as gr
import torch
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.inference import load_model_for_inference, generate_completion

# CSS for code highlighting
code_css = HtmlFormatter().get_style_defs('.highlight')

def format_code(code):
    """Format code with syntax highlighting"""
    highlighted = highlight(code, PythonLexer(), HtmlFormatter())
    return f'<style>{code_css}</style>{highlighted}'

def complete_code(code, max_new_tokens, temperature):
    """Generate code completion"""
    try:
        # Load model and tokenizer (with caching)
        if not hasattr(complete_code, 'model') or not hasattr(complete_code, 'tokenizer'):
            print("Loading model and tokenizer...")
            model_path = os.environ.get("MODEL_PATH", "models/code-completion-model")
            
            # Use base model if fine-tuned model doesn't exist
            if not os.path.exists(model_path):
                model_path = "Salesforce/codegen-350M-mono"
                use_peft = False
            else:
                use_peft = True
                
            complete_code.model, complete_code.tokenizer = load_model_for_inference(
                model_path=model_path,
                use_peft=use_peft
            )
            print("Model loaded successfully!")
        
        # Generate completion
        completion = generate_completion(
            model=complete_code.model,
            tokenizer=complete_code.tokenizer,
            function_prefix=code,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Format the output with syntax highlighting
        return format_code(completion)
        
    except Exception as e:
        return f"<p style='color: red'>Error: {str(e)}</p>"

# Create the Gradio interface
with gr.Blocks(title="AI Code Completion") as demo:
    gr.Markdown("# AI Code Completion Demo")
    gr.Markdown("Enter Python code below and let the AI complete it for you.")
    
    with gr.Row():
        with gr.Column(scale=2):
            code_input = gr.Textbox(
                placeholder="# Enter Python code here\ndef fibonacci(n):\n    # Calculate the nth fibonacci number", 
                label="Code Input", 
                lines=10
            )
            
            with gr.Row():
                max_tokens = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Max New Tokens")
                temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                
            complete_btn = gr.Button("Complete Code")
            
        with gr.Column(scale=2):
            output = gr.HTML(label="Generated Completion")
    
    complete_btn.click(
        fn=complete_code, 
        inputs=[code_input, max_tokens, temperature], 
        outputs=output
    )
    
    # Example code snippets
    examples = [
        ["def fibonacci(n):\n    # Calculate the nth fibonacci number"],
        ["def sort_list(items):\n    # Sort the list using a custom algorithm"],
        ["def process_image(image_path):\n    # Load and preprocess the image\n    import cv2\n    img = "],
        ["# Connect to a database and fetch all users\nimport sqlite3\n\ndef get_all_users():"],
    ]
    
    gr.Examples(examples=examples, inputs=code_input)
    
    gr.Markdown("### About")
    gr.Markdown("""
    This demo uses a fine-tuned Salesforce CodeGen model to generate Python code completions.
    The model was fine-tuned on the Code_Search_Net dataset using parameter-efficient techniques.
    """)

if __name__ == "__main__":
    # Launch the demo
    demo.launch() 