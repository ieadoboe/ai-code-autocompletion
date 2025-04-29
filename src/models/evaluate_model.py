"""
Evaluation metrics for code completion models.
"""
import numpy as np
from typing import Dict, List, Any, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from Levenshtein import distance
import json
import os
import matplotlib.pyplot as plt

def prepare_evaluation_samples(examples, tokenizer, max_tokens=100, truncation_ratio=0.5):
    """
    Prepare evaluation samples from code examples.
    
    Args:
        examples: List of code examples
        tokenizer: Tokenizer to use
        max_tokens: Maximum tokens to consider
        truncation_ratio: Ratio of the code to use as prefix
        
    Returns:
        List of evaluation samples with prefix and expected continuation
    """
    evaluation_samples = []
    
    for example in examples:
        code = example["code"]
        
        # Skip very short functions
        if len(code.strip()) < 20:
            continue
            
        # Tokenize code
        tokens = tokenizer.tokenize(code)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            
        # Split into prefix and expected continuation
        split_idx = int(len(tokens) * truncation_ratio)
        if split_idx < 10:  # Ensure prefix is at least 10 tokens
            continue
            
        prefix_tokens = tokens[:split_idx]
        expected_tokens = tokens[split_idx:]
        
        # Convert back to text
        prefix = tokenizer.convert_tokens_to_string(prefix_tokens)
        expected = tokenizer.convert_tokens_to_string(expected_tokens)
        
        evaluation_samples.append({
            "prefix": prefix,
            "expected": expected,
            "function_name": example.get("function_name", ""),
        })
    
    print(f"Created {len(evaluation_samples)} evaluation samples")
    return evaluation_samples

def compute_bleu_score(expected: str, generated: str) -> float:
    """
    Compute BLEU score for generated code completion.
    
    Args:
        expected: Expected code completion
        generated: Generated code completion
        
    Returns:
        BLEU score
    """
    # Tokenize texts
    reference = word_tokenize(expected)
    candidate = word_tokenize(generated)
    
    # Use smoothing function to avoid zero scores
    smoothie = SmoothingFunction().method1
    
    # Compute BLEU score
    return sentence_bleu([reference], candidate, smoothing_function=smoothie)

def compute_exact_match_ratio(expected: str, generated: str, ngram_size: int = 4) -> float:
    """
    Compute exact match ratio using character n-grams.
    
    Args:
        expected: Expected code completion
        generated: Generated code completion
        ngram_size: Size of character n-grams to compare
        
    Returns:
        Exact match ratio
    """
    # Extract character n-grams
    def get_ngrams(text, n):
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    expected_ngrams = set(get_ngrams(expected, ngram_size))
    generated_ngrams = set(get_ngrams(generated, ngram_size))
    
    if not expected_ngrams:
        return 0.0
    
    # Calculate overlap
    common_ngrams = expected_ngrams.intersection(generated_ngrams)
    return len(common_ngrams) / len(expected_ngrams)

def compute_edit_distance(expected: str, generated: str) -> int:
    """
    Compute Levenshtein edit distance between expected and generated completions.
    
    Args:
        expected: Expected code completion
        generated: Generated code completion
        
    Returns:
        Edit distance (lower is better)
    """
    return distance(expected, generated)

def compute_normalized_edit_distance(expected: str, generated: str) -> float:
    """
    Compute normalized Levenshtein edit distance.
    
    Args:
        expected: Expected code completion
        generated: Generated code completion
        
    Returns:
        Normalized edit distance (lower is better)
    """
    if not expected:
        return 1.0 if generated else 0.0
        
    return distance(expected, generated) / max(len(expected), len(generated))

def evaluate_completions(evaluation_samples: List[Dict], 
                        model: Any, 
                        tokenizer: Any,
                        max_new_tokens: int = 100,
                        output_dir: str = "data/evaluation") -> Dict:
    """
    Evaluate model completions on evaluation samples.
    
    Args:
        evaluation_samples: List of evaluation samples
        model: Model to evaluate
        tokenizer: Tokenizer to use
        max_new_tokens: Maximum new tokens for generation
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    from src.models.inference import generate_completion
    
    results = []
    metrics = {
        "bleu_scores": [],
        "exact_match_ratios": [],
        "edit_distances": [],
        "normalized_edit_distances": [],
    }
    
    for sample in evaluation_samples:
        prefix = sample["prefix"]
        expected = sample["expected"]
        
        # Generate completion
        generated = generate_completion(
            model, 
            tokenizer, 
            prefix, 
            max_new_tokens=max_new_tokens
        )
        
        # Compute metrics
        bleu = compute_bleu_score(expected, generated)
        exact_match = compute_exact_match_ratio(expected, generated)
        edit_dist = compute_edit_distance(expected, generated)
        norm_edit_dist = compute_normalized_edit_distance(expected, generated)
        
        # Update metrics
        metrics["bleu_scores"].append(bleu)
        metrics["exact_match_ratios"].append(exact_match)
        metrics["edit_distances"].append(edit_dist)
        metrics["normalized_edit_distances"].append(norm_edit_dist)
        
        # Save results
        results.append({
            "prefix": prefix,
            "expected": expected,
            "generated": generated,
            "function_name": sample.get("function_name", ""),
            "bleu": bleu,
            "exact_match": exact_match,
            "edit_distance": edit_dist,
            "normalized_edit_distance": norm_edit_dist,
        })
    
    # Compute aggregate metrics
    aggregate_metrics = {
        "bleu_score_mean": np.mean(metrics["bleu_scores"]),
        "bleu_score_std": np.std(metrics["bleu_scores"]),
        "exact_match_ratio_mean": np.mean(metrics["exact_match_ratios"]),
        "exact_match_ratio_std": np.std(metrics["exact_match_ratios"]),
        "edit_distance_mean": np.mean(metrics["edit_distances"]),
        "edit_distance_std": np.std(metrics["edit_distances"]),
        "normalized_edit_distance_mean": np.mean(metrics["normalized_edit_distances"]),
        "normalized_edit_distance_std": np.std(metrics["normalized_edit_distances"]),
        "num_samples": len(evaluation_samples),
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump({
            "results": results,
            "aggregate_metrics": aggregate_metrics,
        }, f, indent=2)
    
    # Return metrics
    return aggregate_metrics

def plot_evaluation_metrics(metrics: Dict, output_dir: str = "data/evaluation"):
    """
    Plot evaluation metrics.
    
    Args:
        metrics: Dictionary with evaluation metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot BLEU score distribution
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # BLEU score
    axes[0, 0].hist(metrics["bleu_scores"], bins=20)
    axes[0, 0].set_title(f"BLEU Score Distribution (Mean: {metrics['bleu_score_mean']:.4f})")
    axes[0, 0].set_xlabel("BLEU Score")
    axes[0, 0].set_ylabel("Count")
    
    # Exact match ratio
    axes[0, 1].hist(metrics["exact_match_ratios"], bins=20)
    axes[0, 1].set_title(f"Exact Match Ratio Distribution (Mean: {metrics['exact_match_ratio_mean']:.4f})")
    axes[0, 1].set_xlabel("Exact Match Ratio")
    axes[0, 1].set_ylabel("Count")
    
    # Edit distance
    axes[1, 0].hist(metrics["edit_distances"], bins=20)
    axes[1, 0].set_title(f"Edit Distance Distribution (Mean: {metrics['edit_distance_mean']:.4f})")
    axes[1, 0].set_xlabel("Edit Distance")
    axes[1, 0].set_ylabel("Count")
    
    # Normalized edit distance
    axes[1, 1].hist(metrics["normalized_edit_distances"], bins=20)
    axes[1, 1].set_title(f"Normalized Edit Distance Distribution (Mean: {metrics['normalized_edit_distance_mean']:.4f})")
    axes[1, 1].set_xlabel("Normalized Edit Distance")
    axes[1, 1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_metrics.png"))
    
    print(f"Saved evaluation plots to {output_dir}")

if __name__ == "__main__":
    # Example usage
    import nltk
    nltk.download('punkt')
    
    from src.data.load_data import load_examples_from_disk
    from src.features.tokenization import get_tokenizer
    from src.models.inference import load_model_for_inference
    
    # Load tokenizer and model
    model_path = "./code-completion-model"
    use_peft = os.path.exists(model_path)
    if not use_peft:
        model_path = "Salesforce/codegen-350M-mono"
    
    model, tokenizer = load_model_for_inference(model_path, use_peft=use_peft)
    
    # Load examples
    examples = load_examples_from_disk()
    
    # Prepare evaluation samples
    evaluation_samples = prepare_evaluation_samples(examples, tokenizer)
    
    # Evaluate completions
    metrics = evaluate_completions(evaluation_samples[:20], model, tokenizer)
    print("Evaluation metrics:", metrics)
    
    # Plot metrics
    plot_evaluation_metrics(metrics) 