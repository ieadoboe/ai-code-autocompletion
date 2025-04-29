#!/bin/bash
# AI Code Autocompletion Pipeline
# This script demonstrates how to run the entire code autocompletion pipeline
# It executes data preparation, model training, evaluation and example completion in sequence

# Exit immediately if any command fails
set -e

# Create directory structure for storing data and models
# Makes nested directories for raw data, processed data, evaluation results
mkdir -p data/raw data/processed data/evaluation
# Makes directory for storing trained models
mkdir -p models/code-completion

# Step 1: Prepare the training data
echo "========== STEP 1: Prepare Data =========="
# Downloads and processes Python function examples from HuggingFace dataset
# Saves 1000 processed examples to JSON file
python main.py prepare-data \
  --dataset ieadoboe/python-function-examples \
  --output data/processed/training_examples.json \
  --max-examples 1000

# Step 2: Train the model
echo "========== STEP 2: Train Model =========="
# Trains model on processed examples for 3 epochs
# Uses batch size of 4 with 4 gradient accumulation steps
python main.py train \
  --data data/processed/training_examples.json \
  --output models/code-completion \
  --epochs 3 \
  --batch-size 4 \
  --grad-accum-steps 4

# Step 3: Evaluate model performance
echo "========== STEP 3: Evaluate Model =========="
# Evaluates trained model on 100 examples from training data
# Saves evaluation metrics and plots to evaluation directory
python main.py evaluate \
  --model models/code-completion \
  --data data/processed/training_examples.json \
  --output data/evaluation \
  --num-samples 100

# Step 4: Run an example code completion
echo "========== STEP 4: Example Completion =========="
# Tests model by completing a sample image processing function
# Generates up to 100 new tokens to complete the function
python main.py complete \
  "def process_image(image_path):\n    # Load and preprocess image\n    import numpy as np\n    img = " \
  --model models/code-completion \
  --max-tokens 100

# Print completion message and instructions for interactive mode
echo "========== Pipeline Completed =========="
echo "Run the interactive mode with:"
# Shows command to start interactive code completion session
echo "python main.py interactive --model models/code-completion"