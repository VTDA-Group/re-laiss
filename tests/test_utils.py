import pytest
import tempfile
import os
import pandas as pd
import numpy as np
import hashlib
import pickle
from pathlib import Path
from relaiss.utils import (
    compute_dataframe_hash,
    get_cache_key,
    get_cache_dir,
    cache_dataframe,
    load_cached_dataframe,
    suppress_output
)

def test_suppress_output():
    """Test that suppress_output context manager works."""
    with suppress_output():
        print("This should not be visible")
    # No assertion needed, just checking it doesn't raise an exception

def test_get_cache_dir():
    """Test that get_cache_dir returns a valid path."""
    cache_dir = get_cache_dir()
    assert isinstance(cache_dir, Path)
    assert cache_dir.name == 'cache'
    assert '.relaiss' in str(cache_dir)

def test_compute_dataframe_hash():
    """Test that compute_dataframe_hash returns a valid hash."""
    # Create a test dataframe
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    
    # Compute hash
    df_hash = compute_dataframe_hash(df)
    
    # Check that hash is a string
    assert isinstance(df_hash, str)
    
    # Check that hash is consistent
    assert compute_dataframe_hash(df) == df_hash
    
    # Check that different dataframes have different hashes
    df2 = pd.DataFrame({
        'a': [1, 2, 4],  # Changed one value
        'b': ['x', 'y', 'z']
    })
    assert compute_dataframe_hash(df2) != df_hash

def test_get_cache_key():
    """Test that get_cache_key returns a valid key."""
    # Test with string
    key = get_cache_key('test')
    assert isinstance(key, str)
    
    # Test with different inputs produce different keys
    key1 = get_cache_key('test', a=1, b='x')
    key2 = get_cache_key('test', a=2, b='x')
    assert key1 != key2
    
    # Test with complex types
    key3 = get_cache_key('test', arr=np.array([1, 2, 3]), df=pd.DataFrame({'a': [1, 2]}))
    assert isinstance(key3, str)

def test_cache_and_load_dataframe():
    """Test caching and loading a dataframe."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up a test environment
        os.environ['HOME'] = tmpdir
        
        # Create a test dataframe
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': ['x', 'y', 'z']
        })
        
        # Generate a cache key
        key = get_cache_key('test_df')
        
        # Cache the dataframe
        cache_dataframe(df, key)
        
        # Load the dataframe
        loaded_df = load_cached_dataframe(key)
        
        # Check that the loaded dataframe is equal to the original
        pd.testing.assert_frame_equal(df, loaded_df)
        
        # Test loading a non-existent key
        assert load_cached_dataframe('nonexistent_key') is None 