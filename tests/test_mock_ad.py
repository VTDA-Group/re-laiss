import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
from relaiss.anomaly import train_AD_model

def test_train_AD_model_with_preprocessed_df():
    """Test training AD model with preprocessed dataframe."""
    # Create a sample preprocessed dataframe
    n_samples = 100
    np.random.seed(42)
    
    df = pd.DataFrame({
        'g_peak_mag': np.random.normal(20, 1, n_samples),
        'r_peak_mag': np.random.normal(19, 1, n_samples),
        'g_peak_time': np.random.uniform(0, 100, n_samples),
        'r_peak_time': np.random.uniform(0, 100, n_samples),
        'host_ra': np.random.uniform(0, 360, n_samples),
        'host_dec': np.random.uniform(-90, 90, n_samples),
        'ztf_object_id': [f'ZTF{i:08d}' for i in range(n_samples)]
    })
    
    # Define feature lists
    lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
    host_features = ['host_ra', 'host_dec']
    
    # Create a temporary directory for models
    with tempfile.TemporaryDirectory() as tmpdir:
        # Train model
        model_path = train_AD_model(
            lc_features=lc_features,
            host_features=host_features,
            preprocessed_df=df,
            path_to_models_directory=tmpdir,
            n_estimators=10,  # Use small value for faster tests
            contamination=0.05,
            max_samples=10,
            force_retrain=True
        )
        
        # Check if model file exists
        assert os.path.exists(model_path)
        assert "IForest" in os.path.basename(model_path)
        assert "n=10" in os.path.basename(model_path)

def test_train_AD_model_error_handling():
    """Test error handling in train_AD_model."""
    # Test with missing required parameters
    with pytest.raises(ValueError):
        train_AD_model(
            lc_features=['g_peak_mag'],
            host_features=['host_ra'],
            preprocessed_df=None,
            path_to_dataset_bank=None
        )

@patch('sklearn.ensemble.IsolationForest')
def test_train_AD_model_with_joblib(mock_iforest):
    """Test train_AD_model with joblib mocked."""
    # Create mock model
    mock_model = MagicMock()
    mock_iforest.return_value = mock_model
    
    # Create sample data
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('joblib.dump') as mock_dump:
            # Call train_AD_model
            model_path = train_AD_model(
                lc_features=['feature1'],
                host_features=['feature2'],
                preprocessed_df=df,
                path_to_models_directory=tmpdir,
                force_retrain=True
            )
            
            # Check that joblib.dump was called with the right arguments
            mock_dump.assert_called_once()
            assert mock_dump.call_args[0][0] == mock_model
            assert mock_dump.call_args[0][1] == model_path
            
            # Check that fit was called with the right data
            assert mock_model.fit.called
            fit_arg = mock_model.fit.call_args[0][0]
            assert fit_arg.shape == (3, 2)  # 3 rows, 2 columns 