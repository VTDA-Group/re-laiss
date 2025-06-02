import pytest
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler
from relaiss.relaiss import ReLAISS

def test_anomaly_detection_simplified(tmp_path):
    """Test anomaly detection with minimal dependencies."""
    from relaiss.anomaly import anomaly_detection
    
    # Create necessary directories
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir(exist_ok=True)
    (figure_dir / "AD").mkdir(exist_ok=True)
    
    # Create model data with scaler and training features
    scaler = StandardScaler()
    X = np.random.rand(20, 4)
    scaler.fit(X)
    
    # Define features first
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    client = ReLAISS()
    client.load_reference()
    client.built_for_AD = True  # Set this flag to use preprocessed_df
    
    model_path = model_dir / "kNN_scaler_lc=2_host=2.pkl"
    
    model_data = {
        'scaler': scaler,
        'training_features': X,
        'feature_names': lc_features + host_features
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Create preprocessed dataframe
    df = pd.DataFrame({
        'g_peak_mag': np.random.normal(20, 1, 100),
        'r_peak_mag': np.random.normal(19, 1, 100),
        'host_ra': np.random.uniform(0, 360, 100),
        'host_dec': np.random.uniform(-90, 90, 100),
    })
    
    # Mock functions
    with patch('relaiss.anomaly.check_anom_and_plot') as mock_check_anom, \
         patch('relaiss.anomaly.train_AD_model', return_value=str(model_path)), \
         patch('relaiss.anomaly.get_timeseries_df') as mock_ts, \
         patch('relaiss.anomaly.get_TNS_data', return_value=("TestSN", "Ia", 0.1)), \
         patch('relaiss.features.build_dataset_bank') as mock_build_bank, \
         patch('antares_client.search.get_by_ztf_object_id') as mock_antares:
        
        # Configure mocks
        # Store the results so we can return them from our test
        anomaly_results = {
            'anomaly_scores': np.random.uniform(0, 1, 10),
            'anomaly_labels': np.random.choice([0, 1], size=10)
        }
        
        # Side effect function to capture the result and return it for our test
        def side_effect(*args, **kwargs):
            # Return mock values that match the expected unpacking
            return np.array([58000.0]), np.array([0.5]), np.array([0.75])
        
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
        
        # Configure the mock ANTARES client
        mock_locus = MagicMock()
        mock_ts = MagicMock()
        mock_ts.to_pandas.return_value = pd.DataFrame({
            'ant_mjd': np.linspace(0, 100, 10),
            'ant_passband': ['g', 'r'] * 5,
            'ant_mag': np.random.normal(20, 0.5, 10),
            'ant_magerr': np.random.uniform(0.01, 0.1, 10),
            'ant_ra': [150.0] * 10,
            'ant_dec': [20.0] * 10
        })
        mock_locus.timeseries = mock_ts
        mock_locus.catalog_objects = {
            "tns_public_objects": [
                {"name": "TestSN", "type": "Ia", "redshift": 0.1}
            ]
        }
        mock_antares.return_value = mock_locus
        
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
            client=client,
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=str(tmp_path),
            path_to_sfd_folder=None,  # This will be ignored due to our mocking
            path_to_dataset_bank=None,  # Should be None when using preprocessed_df
            path_to_models_directory=str(model_dir),
            path_to_figure_directory=str(figure_dir),
            save_figures=True,
            force_retrain=False,
            preprocessed_df=df,  # Use preprocessed_df
            return_scores=True  # Set to True to get the return values
        )
        
        # Verify check_anom_and_plot was called
        mock_check_anom.assert_called_once()
        
        # Verify the return values
        assert result is not None
        mjd_anom, anom_scores, norm_scores = result
        assert isinstance(mjd_anom, np.ndarray)
        assert isinstance(anom_scores, np.ndarray)
        assert isinstance(norm_scores, np.ndarray) 
