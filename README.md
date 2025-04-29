# AI-powered Code Autocompletion

This project demonstrates the implementation of an AI-powered code autocompletion model using the Salesforce CodeGen model with parameter-efficient fine-tuning techniques.

## Project Overview

Code autocompletion tools have revolutionized software development by predicting and suggesting code as developers type. The growing insurgence of GitHub Copilot and Cursor Tab, a code autocomplete project was simply a must-try for me. This project aims to fine-tune a pre-trained Large Language Model (LLM) to complete Python code snippets based on function signatures, docstrings and the function itself. The model is trained on Python functions from the `Code_Search_Net` dataset.

## Features

- Uses the [`Salesforce/codegen-350M-mono`](https://huggingface.co/Salesforce/codegen-350M-mono) model as a base model (from `huggingface`)
- Implements Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) to reduce trainable parameters
- Training and evaluation pipeline for code completion
- Tested with some example generation functionality

## Technical Implementation

The project includes:

1. Data preparation from the `Code_Search_Net` dataset
2. Tokenization optimized for code
3. Parameter-Efficient Fine-Tuning with LoRA
4. Training and evaluation pipeline
5. Example generation and testing

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- PEFT library

All project libraries and package requirements are listed in the `requirements.txt` file. To install:

```bash
pip install [package-name]
```

or in conda environments (recommended):

```bash
conda install [package-name]
```

### Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai-code-autocompletion.git
   cd ai-code-autocompletion
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook:

   ```bash
   jupyter notebook notebook/ai_code_autocompletion.ipynb
   ```

4. Follow the steps in the notebook to train and evaluate the model

## Example Usage

After training, you can use the model to complete code snippets:

```python
# Example code completion
function_prefix = "def process_image(image_path):\n    # Load and preprocess image\n    import numpy as np\n    img = "
completion = generate_completion(model, tokenizer, function_prefix)
print(f"Completion: {completion}")
```

## Challenges and Limitations

1. Context Window Limitations: The model can only see a limited amount of context (often just the current function), making it difficult to understand the broader codebase.
2. Computational Efficiency: Code suggestions must appear nearly instantaneously to be useful, requiring model optimization for low latency.

## Conclusion

This project demonstrates the feasibility of fine-tuning a code autocompletion model using modern techniques such as PEFT and LoRA. The Salesforce CodeGen model shows promising results for Python code completion tasks.

## Remarks

This (the notebook in `notebook/`) was my original submission for my end semester project for the Deep Learning and AI course I took (Winter 2025). I decided to expand the idea into this fully-flegded project you see now using industry best practices for model development and test. This is the first of its kind for me. If you're reading this, it really means a lot to me.
