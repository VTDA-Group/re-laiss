import pytest
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.ensemble import IsolationForest

def test_train_AD_model_simple(tmp_path):
    """Test training AD model with simplified mocks."""
    from relaiss.anomaly import train_AD_model
    
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    df = pd.DataFrame({
        'g_peak_mag': np.random.normal(20, 1, 100),
        'r_peak_mag': np.random.normal(19, 1, 100),
        'host_ra': np.random.uniform(0, 360, 100),
        'host_dec': np.random.uniform(-90, 90, 100),
    })
    
    with patch('sklearn.ensemble.IsolationForest', autospec=True) as mock_iso:
        mock_model = MagicMock()
        mock_model.n_estimators = 100
        mock_model.contamination = 0.02
        mock_model.max_samples = 256
        mock_iso.return_value = mock_model
        
        with patch('joblib.dump') as mock_dump:
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
            
            assert model_path.endswith('.pkl') or model_path.endswith('.joblib')
            mock_dump.assert_called_once()
            
            model_arg = mock_dump.call_args[0][0]
            assert model_arg.n_estimators == 100
            assert model_arg.contamination == 0.02
            assert model_arg.max_samples == 256

def test_anomaly_detection_simplified(tmp_path):
    """Test anomaly detection with minimal dependencies."""
    from relaiss.anomaly import anomaly_detection
    
    # Create necessary directories
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir(exist_ok=True)
    (figure_dir / "AD").mkdir(exist_ok=True)
    
    # Create IsolationForest model
    real_forest = IsolationForest(n_estimators=10, random_state=42)
    X = np.random.rand(20, 4)
    real_forest.fit(X)
    
    model_path = model_dir / "IForest_n=100_c=0.02_m=256.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(real_forest, f)
    
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Mock functions
    with patch('relaiss.anomaly.check_anom_and_plot') as mock_check_anom, \
         patch('relaiss.anomaly.train_AD_model', return_value=str(model_path)), \
         patch('relaiss.anomaly.get_timeseries_df') as mock_ts, \
         patch('relaiss.anomaly.get_TNS_data', return_value=("TestSN", "Ia", 0.1)), \
         patch('relaiss.features.build_dataset_bank') as mock_build_bank:
        
        # Configure mocks
        # Store the results so we can return them from our test
        anomaly_results = {
            'anomaly_scores': np.random.uniform(0, 1, 10),
            'anomaly_labels': np.random.choice([0, 1], size=10)
        }
        
        # Side effect function to capture the result and return it for our test
        def side_effect(*args, **kwargs):
            return None  # Original function returns None
            
        mock_check_anom.side_effect = side_effect
        
        mock_ts.return_value = pd.DataFrame({
            'mjd': np.linspace(58000, 58050, 10),
            'mag': np.random.normal(20, 0.5, 10),
            'magerr': np.random.uniform(0.01, 0.1, 10),
            'band': ['g', 'r'] * 5,
            'g_peak_mag': [20.0] * 10,
            'r_peak_mag': [19.5] * 10,
            'host_ra': [150.0] * 10,
            'host_dec': [20.0] * 10,
            'mjd_cutoff': np.linspace(58000, 58050, 10),
            'obs_num': list(range(1, 11))
        })
        
        # Mock build_dataset_bank to avoid SFD file access
        features_df = pd.DataFrame({
            'ztf_object_id': ['ZTF21abbzjeq'],
            'g_peak_mag': [20.0],
            'r_peak_mag': [19.5],
            'host_ra': [150.0],
            'host_dec': [20.0]
        })
        mock_build_bank.return_value = features_df
        
        # Run the function
        result = anomaly_detection(
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=str(tmp_path),
            path_to_sfd_folder=None,  # This will be ignored due to our mocking
            path_to_dataset_bank=None,  # This will be ignored due to our mocking
            path_to_models_directory=str(model_dir),
            path_to_figure_directory=str(figure_dir),
            save_figures=True,
            force_retrain=False
        )
        
        # Verify the mock was called
        mock_check_anom.assert_called_once()
        
        # Should return None since that's what the function is defined to return
        assert result is None 