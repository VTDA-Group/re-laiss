import pytest
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
from relaiss.anomaly import train_AD_model
from sklearn.ensemble import IsolationForest

def test_train_AD_model_simple(tmp_path):
    """Test training AD model with simplified mocks."""
    # Create a simple DataFrame with the necessary features
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Mock preprocessed dataframe
    df = pd.DataFrame({
        'g_peak_mag': np.random.normal(20, 1, 100),
        'r_peak_mag': np.random.normal(19, 1, 100),
        'host_ra': np.random.uniform(0, 360, 100),
        'host_dec': np.random.uniform(-90, 90, 100),
    })
    
    # Mock the IsolationForest and joblib
    with patch('sklearn.ensemble.IsolationForest', autospec=True) as mock_iso:
        mock_model = MagicMock()
        mock_model.n_estimators = 100
        mock_model.contamination = 0.02
        mock_model.max_samples = 256
        mock_iso.return_value = mock_model
        
        with patch('joblib.dump') as mock_dump:
            # Run the function
            model_path = train_AD_model(
                lc_features=lc_features,
                host_features=host_features,
                preprocessed_df=df,
                path_to_models_directory=str(tmp_path),
                n_estimators=100,
                contamination=0.02,
                max_samples=256,
                force_retrain=True
            )
            
            # Don't check exact filename - it depends on the implementation
            # Instead check that it ends with .pkl or .joblib
            assert model_path.endswith('.pkl') or model_path.endswith('.joblib')
            
            # Verify joblib.dump was called
            mock_dump.assert_called_once()
            
            # Extract the model from the call
            model_arg = mock_dump.call_args[0][0]
            assert model_arg.n_estimators == 100
            assert model_arg.contamination == 0.02
            assert model_arg.max_samples == 256

def test_anomaly_detection_simplified(tmp_path):
    """Test anomaly detection with minimal dependencies."""
    from relaiss.anomaly import anomaly_detection
    
    # Create the necessary directories
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir(exist_ok=True)
    (figure_dir / "AD").mkdir(exist_ok=True)
    
    # Create a dummy model file with actual content
    model_path = model_dir / "IForest_n=100_c=0.02_m=256.pkl"  # Use actual filename format
    
    # Create a simple isolation forest and save it to the file
    mock_if = MagicMock()
    mock_if.predict.return_value = np.ones(10)  # All normal
    mock_if.decision_function.return_value = np.random.uniform(0, 1, 10)  # Random scores
    
    # Write actual data to the file
    with open(model_path, 'wb') as f:
        pickle.dump(mock_if, f)
    
    # Features to test with
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Mock functions and classes
    with patch('relaiss.anomaly.train_AD_model', return_value=str(model_path)), \
         patch('relaiss.anomaly.get_timeseries_df') as mock_ts, \
         patch('relaiss.anomaly.get_TNS_data', return_value=("TestSN", "Ia", 0.1)), \
         patch('matplotlib.pyplot.figure', return_value=MagicMock()), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('matplotlib.pyplot.show'):
        
        # Configure mock timeseries
        mock_ts.return_value = pd.DataFrame({
            'mjd': np.linspace(58000, 58050, 10),
            'mag': np.random.normal(20, 0.5, 10),
            'magerr': np.random.uniform(0.01, 0.1, 10),
            'band': ['g', 'r'] * 5,
            'g_peak_mag': [20.0] * 10,
            'r_peak_mag': [19.5] * 10,
            'host_ra': [150.0] * 10,
            'host_dec': [20.0] * 10,
        })
        
        # Run anomaly detection with minimal config
        result = anomaly_detection(
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=str(tmp_path),
            path_to_sfd_data_folder=None,  # Not needed with our mocks
            path_to_dataset_bank=None,  # Not needed with our mocks
            path_to_models_directory=str(model_dir),
            path_to_figure_directory=str(figure_dir),
            save_figures=True,
            force_retrain=False
        )
        
        # Verify the result
        assert isinstance(result, dict)
        assert "anomaly_scores" in result
        assert "anomaly_labels" in result

# Updated test that uses a real IsolationForest
def test_anomaly_detection_simplified(tmp_path):
    """Test anomaly detection with minimal dependencies."""
    from relaiss.anomaly import anomaly_detection
    
    # Create the necessary directories
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir(exist_ok=True)
    (figure_dir / "AD").mkdir(exist_ok=True)
    
    # Create a real IsolationForest that can be pickled
    real_forest = IsolationForest(n_estimators=10, random_state=42)
    X = np.random.rand(20, 4)  # Some dummy data
    real_forest.fit(X)  # Fit the model so it can be used
    
    # Create a dummy model file with actual content
    model_path = model_dir / "IForest_n=100_c=0.02_m=256.pkl"  # Use actual filename format
    
    # Write the real forest to the file
    with open(model_path, 'wb') as f:
        pickle.dump(real_forest, f)
    
    # Features to test with - should match the dimensions used to create X above
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Mock functions and classes
    with patch('relaiss.anomaly.train_AD_model', return_value=str(model_path)), \
         patch('relaiss.anomaly.get_timeseries_df') as mock_ts, \
         patch('relaiss.anomaly.get_TNS_data', return_value=("TestSN", "Ia", 0.1)), \
         patch('matplotlib.pyplot.figure', return_value=MagicMock()), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('matplotlib.pyplot.show'):
        
        # Configure mock timeseries with the 4 features we need
        mock_ts.return_value = pd.DataFrame({
            'mjd': np.linspace(58000, 58050, 10),
            'mag': np.random.normal(20, 0.5, 10),
            'magerr': np.random.uniform(0.01, 0.1, 10),
            'band': ['g', 'r'] * 5,
            'g_peak_mag': [20.0] * 10,
            'r_peak_mag': [19.5] * 10,
            'host_ra': [150.0] * 10,
            'host_dec': [20.0] * 10,
        })
        
        # Run anomaly detection function
        result = anomaly_detection(
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=str(tmp_path),
            path_to_sfd_data_folder=None,  # Not needed with our mocks
            path_to_dataset_bank=None,  # Not needed with our mocks
            path_to_models_directory=str(model_dir),
            path_to_figure_directory=str(figure_dir),
            save_figures=True,
            force_retrain=False
        )
        
        # Verify the result
        assert isinstance(result, dict)
        assert "anomaly_scores" in result
        assert "anomaly_labels" in result 