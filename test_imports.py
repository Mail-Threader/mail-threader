"""
Test script to verify that imports work correctly with the new directory structure.
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Import from the main modules
    from src.data_preparation import DataPreparation
    from src.story_development import StoryDevelopment
    from src.summarization_classification import SummarizationClassification
    from src.visualization import Visualization
    from src.utils import sample_function, safe_get, filter_list
    
    print("All imports successful!")
    
    # Test a utility function
    result = sample_function(1, 2)
    print(f"sample_function(1, 2) = {result}")
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)