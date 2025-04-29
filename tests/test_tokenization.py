"""
Tests for the tokenization module.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.tokenization import (
    get_tokenizer, 
    prepare_training_samples, 
    CodeCompletionDataset,
    split_train_val
)

class TestTokenization(unittest.TestCase):
    """Test cases for tokenization module."""
    
    def test_get_tokenizer(self):
        """Test that the tokenizer is loaded correctly."""
        with patch('src.features.tokenization.AutoTokenizer') as mock_tokenizer:
            # Setup mock
            mock_instance = MagicMock()
            mock_instance.pad_token_id = None
            mock_instance.eos_token_id = 50256
            mock_tokenizer.from_pretrained.return_value = mock_instance
            
            # Call the function
            tokenizer = get_tokenizer()
            
            # Assert that AutoTokenizer.from_pretrained was called with the right args
            mock_tokenizer.from_pretrained.assert_called_once_with("Salesforce/codegen-350M-mono")
            
            # Assert that the pad_token_id was set
            self.assertEqual(tokenizer.pad_token_id, 50256)
    
    def test_prepare_training_samples(self):
        """Test preparing training samples."""
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": list(range(50))}  # 50 tokens
        
        # Example data
        examples = [
            {"code": "def function(): pass", "function_name": "function"},
            {"code": "def long_function():\n    # This is a long function\n    x = 1\n    y = 2\n    return x + y", "function_name": "long_function"},
            {"code": "a", "function_name": "too_short"}  # Should be skipped
        ]
        
        # Call the function
        with patch('random.uniform', return_value=0.7):  # Fix the random value for testing
            samples = prepare_training_samples(examples, mock_tokenizer)
        
        # Assertions
        self.assertGreater(len(samples), 0)
        self.assertIn("input_ids", samples[0])
        self.assertIn("labels", samples[0])
    
    def test_split_train_val(self):
        """Test splitting samples into training and validation sets."""
        # Sample data
        samples = [{"id": i} for i in range(100)]
        
        # Call the function
        train_samples, val_samples = split_train_val(samples, val_ratio=0.2)
        
        # Assertions
        self.assertEqual(len(train_samples), 80)
        self.assertEqual(len(val_samples), 20)
        self.assertEqual(train_samples[0]["id"], 0)
        self.assertEqual(val_samples[0]["id"], 80)

if __name__ == '__main__':
    unittest.main() 