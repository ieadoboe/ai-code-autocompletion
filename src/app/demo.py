"""
Demo application for AI Code Completion using Streamlit.
This provides a simple web interface for the code completion model.
"""
# Fix for Streamlit-PyTorch compatibility issue
import os
import sys

# Set environment variables to help with PyTorch compatibility
os.environ["STREAMLIT_WATCHDOG_PROCESS_ONE_AT_A_TIME"] = "true"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import torch

# Import after setting environment variables
from src.models.inference import load_model_for_inference, generate_completion

# Page configuration
st.set_page_config(
    page_title="AI Code Completion",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Style for main action button */
    button[kind="primary"] {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: 1px solid #4CAF50;
    }
    button[kind="primary"]:hover {
        background-color: #349F37;
        border: 1px solid #349F37;
    }
    
    /* Style for example buttons */
    button[kind="tertiary"] {
        color: #B8B8B8;
        border: 1px solid #B8B8B8;  
        font-weight: normal;
    }
    button[kind="tertiary"]:hover {
        color: white;
    }
    
    /* Style for copy button */
    button[kind="secondary"] {
        background-color: #3498DB; 
        color: white;
        border: 1px solid #3498DB;
    }
    /* Style for copy button */
    button[kind="secondary"]:hover {
        background-color: #299EEC;
        color: white;
        border: 1px solid #299EEC;
    }
    
    /* Headers */
    .code-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'code_input' not in st.session_state:
    st.session_state.code_input = "# Enter Python code here\ndef fibonacci(n):\n    # Calculate the nth fibonacci number"
if 'completion' not in st.session_state:
    st.session_state.completion = ""
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False

# Function to load the model and tokenizer
def load_model():
    try:
        with st.spinner("Loading model and tokenizer... This may take a moment."):
            model_path = os.environ.get("MODEL_PATH", "models/code-completion-model")
            
            # Use base model if fine-tuned model doesn't exist
            if not os.path.exists(model_path):
                model_path = "Salesforce/codegen-350M-mono"
                use_peft = False
            else:
                use_peft = True
                
            st.session_state.model, st.session_state.tokenizer = load_model_for_inference(
                model_path=model_path,
                use_peft=use_peft
            )
            return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False

# Function to generate completions
def complete_code():
    # Check if we need to load the model
    if st.session_state.model is None or st.session_state.tokenizer is None:
        success = load_model()
        if not success:
            return
    
    st.session_state.is_generating = True
    
    try:
        # Generate completion
        completion = generate_completion(
            model=st.session_state.model,
            tokenizer=st.session_state.tokenizer,
            function_prefix=st.session_state.code_input,
            max_new_tokens=st.session_state.max_tokens,
            temperature=st.session_state.temperature
        )
        
        st.session_state.completion = completion
    except Exception as e:
        st.error(f"Error generating completion: {str(e)}")
    finally:
        st.session_state.is_generating = False

# Function to update code input when example is selected
def set_example(example):
    st.session_state.code_input = example

# Main app layout
def main():
    # App header
    st.title("AI Code Completion Demo")
    st.markdown("Enter Python code below and let the AI complete it for you.")
    
    # Sidebar with parameters
    with st.sidebar:
        st.header("Parameters")
        st.session_state.max_tokens = st.slider(
            "Max New Tokens", 
            min_value=10, 
            max_value=200, 
            value=50, 
            step=10,
            help="Maximum number of tokens to generate"
        )
        st.session_state.temperature = st.slider(
            "Temperature", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.7, 
            step=0.1,
            help="Higher values produce more diverse outputs"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This demo uses a fine-tuned Salesforce CodeGen model to generate Python code completions.
        The model was fine-tuned on the Code_Search_Net dataset using parameter-efficient techniques.
        """)
    
    # Main area with code input and output
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="code-header">Code Input</p>', unsafe_allow_html=True)
        code_input = st.text_area(
            label="Python Code Input",
            value=st.session_state.code_input,
            height=300,
            key="code_input",
            label_visibility="collapsed",
            placeholder="Enter your Python code here..."
        )
        
        # Complete code button
        if st.button("Complete Code", use_container_width=True, type="primary"):
            complete_code()
        
        # Example buttons
        st.markdown('<p class="code-header">Examples</p>', unsafe_allow_html=True)
        examples = [
            "def fibonacci(n):\n    # Calculate the nth fibonacci number",
            "def sort_list(items):\n    # Sort the list using a custom algorithm",
            "def process_image(image_path):\n    # Load and preprocess the image\n    import cv2\n    img = ",
            "# Connect to a database and fetch all users\nimport sqlite3\n\ndef get_all_users():"
        ]
        
        # Example buttons per row
        buttons_per_row = 1
        for i in range(0, len(examples), buttons_per_row):
            cols = st.columns(buttons_per_row)
            for j in range(buttons_per_row):
                idx = i + j
                if idx < len(examples):
                    # Use a short version of the example for the button text
                    button_text = examples[idx].split('\n')[0][:30] + "..." if len(examples[idx].split('\n')[0]) > 30 else examples[idx].split('\n')[0]
                    cols[j].button(
                        button_text, 
                        key=f"example_{idx}",
                        on_click=set_example,
                        args=(examples[idx],),
                        use_container_width=True,
                        type="tertiary"
                    )
    
    with col2:
        # Display loading spinner during generation
        if st.session_state.is_generating:
            with st.spinner("Generating code completion..."):
                st.empty()
        
        # Show completion if available
        if st.session_state.completion:
            st.markdown('<p class="code-header">Generated Completion</p>', unsafe_allow_html=True)
            st.code(st.session_state.completion, language="python")
            
            st.markdown('<p class="code-header">Full Code</p>', unsafe_allow_html=True)
            full_code = st.session_state.code_input + st.session_state.completion
            st.code(full_code, language="python")
            
            # Add a button to copy the full code
            if st.button("Copy Full Code to Clipboard", use_container_width=True, type="secondary"):
                st.toast("Code copied to clipboard!", icon="✅")
        else:
            st.info("Click 'Complete Code' to generate a completion for your code snippet.")

if __name__ == "__main__":
    main() 