import pytest
import pandas as pd
import numpy as np
import os
import joblib
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock
from relaiss.anomaly import train_AD_model
from sklearn.preprocessing import StandardScaler
import relaiss as relaiss

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

    client = relaiss.ReLAISS()
    client.load_reference()
    client.built_for_AD = True  # Set this flag to use preprocessed_df

    # Create a mock scaler
    mock_scaler = StandardScaler()
    X = np.random.rand(100, 4)
    mock_scaler.fit(X)
    
    with patch('joblib.dump') as mock_dump:
        # Make mock_dump create empty files
        def side_effect(model, path, *args, **kwargs):
            Path(path).touch()
        mock_dump.side_effect = side_effect
        
        model_path = train_AD_model(
            client=client,
            lc_features=lc_features,
            host_features=host_features,
            preprocessed_df=df,
            path_to_models_directory=str(tmp_path),
            force_retrain=True
        )
        
        # Verify the model was saved with the correct filename
        expected_filename = f"kNN_scaler_lc=2_host=2.pkl"
        assert model_path.endswith(expected_filename)
        
        # Verify joblib.dump was called once (only for the model data)
        assert mock_dump.call_count == 1
        
        # The call should be to save the model data dict
        call_args = mock_dump.call_args_list[0][0]
        assert isinstance(call_args[0], dict)
        assert 'scaler' in call_args[0]
        assert call_args[1] == model_path

def test_anomaly_detection_simplified(tmp_path):
    """Test anomaly detection with minimal dependencies."""
    from relaiss.anomaly import anomaly_detection
    
    # Create the necessary directories
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir(exist_ok=True)
    (figure_dir / "AD").mkdir(exist_ok=True)
    
    # Create model data with scaler and training features
    scaler = StandardScaler()
    X = np.random.rand(20, 4)  # Some dummy data
    scaler.fit(X)  # Fit the scaler
    
    # Create a dummy model file with actual content
    model_path = model_dir / "kNN_scaler_lc=2_host=2.pkl"  # Use actual filename format
    
    model_data = {
        'scaler': scaler,
        'training_features': X,
        'feature_names': ['g_peak_mag', 'r_peak_mag', 'host_ra', 'host_dec']
    }
    
    # Write the model data to the file
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
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
         patch('matplotlib.pyplot.show'), \
         patch('relaiss.anomaly.antares_client.search.get_by_ztf_object_id') as mock_antares, \
         patch('relaiss.anomaly.check_anom_and_plot') as mock_check_anom:
        
        # Configure mock timeseries with the 4 features we need and required columns
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
        
        # Configure mock ANTARES client
        mock_locus = MagicMock()
        mock_ts_df = MagicMock()
        mock_ts_df.to_pandas.return_value = pd.DataFrame({
            'ant_mjd': np.linspace(58000, 58050, 10),
            'ant_passband': ['g', 'r'] * 5,
            'ant_mag': np.random.normal(20, 0.5, 10),
            'ant_magerr': np.random.uniform(0.01, 0.1, 10),
            'ant_ra': [150.0] * 10,
            'ant_dec': [20.0] * 10
        })
        mock_locus.timeseries = mock_ts_df
        mock_locus.catalog_objects = {
            "tns_public_objects": [
                {"name": "TestSN", "type": "Ia", "redshift": 0.1}
            ]
        }
        mock_antares.return_value = mock_locus
        
        # Mock check_anom_and_plot to return a tuple of three values
        mock_check_anom.return_value = (
            np.array([58000, 58010, 58020]),  # mjd_anom
            np.array([0.1, 0.2, 0.3]),  # anom_scores
            np.array([0.4, 0.5, 0.6])  # norm_scores
        )

        client = relaiss.ReLAISS()
        client.load_reference()
        
        # Run anomaly detection function
        result = anomaly_detection(
            client=client,
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=str(tmp_path),
            path_to_sfd_folder=None,  # Not needed with our mocks
            path_to_dataset_bank=None,  # Not needed with our mocks
            path_to_models_directory=str(model_dir),
            path_to_figure_directory=str(figure_dir),
            save_figures=True,
            force_retrain=False
        )
        
        # Verify check_anom_and_plot was called
        mock_check_anom.assert_called_once()
        
        # anomaly_detection function now returns None
        assert result is None 
