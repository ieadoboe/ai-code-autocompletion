# Notebooks

This directory contains Jupyter notebooks used for exploration and development.

## Contents

- `ai_code_autocompletion.ipynb`: The original notebook containing the prototype code autocompletion model implementation. This notebook demonstrates the core concepts and techniques used in the project, including:
  - Loading and preprocessing the Code_Search_Net dataset
  - Tokenization for code completion
  - Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA)
  - Model training and evaluation
  - Example code completions

## Relationship to Project Structure

The code from the notebook has been refactored into a proper Python package structure in the `src` directory. The notebook can be used as a reference for the original implementation, but the code in `src` should be used for actual development and execution.

Main improvements in the package structure:

1. Modular components with clear responsibility separation
2. Better error handling and logging
3. Command-line interface for all functionality
4. Proper testing infrastructure
5. More comprehensive evaluation metrics
