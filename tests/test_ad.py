import pytest
import pandas as pd
import numpy as np
import relaiss as rl
from relaiss.anomaly import anomaly_detection, train_AD_model
import os
import joblib
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

@pytest.fixture
def sample_preprocessed_df():
    """Create a sample preprocessed dataframe for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample data with all required columns from constants.py
    df = pd.DataFrame({
        # Core LC features
        'g_peak_mag': np.random.normal(20, 1, n_samples),
        'r_peak_mag': np.random.normal(19, 1, n_samples),
        'g_peak_time': np.random.uniform(0, 100, n_samples),
        'r_peak_time': np.random.uniform(0, 100, n_samples),
        'g_rise_time': np.random.uniform(10, 20, n_samples),
        'g_decline_time': np.random.uniform(20, 40, n_samples),
        'r_rise_time': np.random.uniform(10, 20, n_samples),
        'r_decline_time': np.random.uniform(20, 40, n_samples),
        'g_duration_above_half_flux': np.random.uniform(30, 60, n_samples),
        'r_duration_above_half_flux': np.random.uniform(30, 60, n_samples),
        
        # Amplitude and variability
        'g_amplitude': np.random.uniform(1, 5, n_samples),
        'r_amplitude': np.random.uniform(1, 5, n_samples),
        'g_skewness': np.random.uniform(-2, 2, n_samples),
        'r_skewness': np.random.uniform(-2, 2, n_samples),
        'g_beyond_2sigma': np.random.uniform(0, 1, n_samples),
        'r_beyond_2sigma': np.random.uniform(0, 1, n_samples),
        
        # Color features
        'mean_g-r': np.random.uniform(0.1, 1.0, n_samples),
        'g-r_at_g_peak': np.random.uniform(0.1, 1.0, n_samples),
        'mean_color_rate': np.random.uniform(-0.05, 0.05, n_samples),
        
        # Peak structure features
        'g_n_peaks': np.random.randint(1, 5, n_samples),
        'r_n_peaks': np.random.randint(1, 5, n_samples),
        'g_dt_main_to_secondary_peak': np.random.uniform(0, 50, n_samples),
        'r_dt_main_to_secondary_peak': np.random.uniform(0, 50, n_samples),
        'g_dmag_secondary_peak': np.random.uniform(0.1, 2, n_samples),
        'r_dmag_secondary_peak': np.random.uniform(0.1, 2, n_samples),
        'g_secondary_peak_prominence': np.random.uniform(0.1, 1, n_samples),
        'r_secondary_peak_prominence': np.random.uniform(0.1, 1, n_samples),
        'g_secondary_peak_width': np.random.uniform(1, 10, n_samples),
        'r_secondary_peak_width': np.random.uniform(1, 10, n_samples),
        
        # Rolling variance features
        'g_max_rolling_variance': np.random.uniform(0.001, 0.2, n_samples),
        'r_max_rolling_variance': np.random.uniform(0.001, 0.2, n_samples),
        'g_mean_rolling_variance': np.random.uniform(0.001, 0.1, n_samples),
        'r_mean_rolling_variance': np.random.uniform(0.001, 0.1, n_samples),
        
        # Local curvature features
        'g_rise_local_curvature': np.random.uniform(-0.1, 0.1, n_samples),
        'g_decline_local_curvature': np.random.uniform(-0.1, 0.1, n_samples),
        'r_rise_local_curvature': np.random.uniform(-0.1, 0.1, n_samples),
        'r_decline_local_curvature': np.random.uniform(-0.1, 0.1, n_samples),
        
        # Host position
        'host_ra': np.random.uniform(0, 360, n_samples),
        'host_dec': np.random.uniform(-90, 90, n_samples),
        'ra': np.random.uniform(0, 360, n_samples),
        'dec': np.random.uniform(-90, 90, n_samples),
        
        # Raw host features
        'gKronMag': np.random.normal(21, 0.5, n_samples),
        'rKronMag': np.random.normal(20, 0.5, n_samples),
        'iKronMag': np.random.normal(19.5, 0.5, n_samples),
        'zKronMag': np.random.normal(19, 0.5, n_samples),
        'gKronMagErr': np.random.uniform(0.01, 0.1, n_samples),
        'rKronMagErr': np.random.uniform(0.01, 0.1, n_samples),
        'iKronMagErr': np.random.uniform(0.01, 0.1, n_samples),
        'zKronMagErr': np.random.uniform(0.01, 0.1, n_samples),
        'gKronRad': np.random.uniform(1, 10, n_samples),
        'rKronRad': np.random.uniform(1, 10, n_samples),
        'iKronRad': np.random.uniform(1, 10, n_samples),
        'zKronRad': np.random.uniform(1, 10, n_samples),
        'gExtNSigma': np.random.uniform(1, 5, n_samples),
        'rExtNSigma': np.random.uniform(1, 5, n_samples),
        'iExtNSigma': np.random.uniform(1, 5, n_samples),
        'zExtNSigma': np.random.uniform(1, 5, n_samples),
        'rmomentXX': np.random.uniform(0.5, 1.5, n_samples),
        'rmomentYY': np.random.uniform(0.5, 1.5, n_samples),
        'rmomentXY': np.random.uniform(-0.5, 0.5, n_samples),
        'ztf_object_id': [f'ZTF{i:08d}' for i in range(n_samples)]
    })
    
    # Add some anomalies
    anomaly_idx = np.random.choice(n_samples, size=20, replace=False)
    df.loc[anomaly_idx, 'g_peak_mag'] += 5  # Make these much brighter
    df.loc[anomaly_idx, 'r_peak_mag'] += 5
    
    return df

def test_train_AD_model_with_preprocessed_df(tmp_path, sample_preprocessed_df):
    """Test that train_AD_model correctly uses a provided preprocessed dataframe."""
    from relaiss.anomaly import train_AD_model
    
    # Define features to use
    lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
    host_features = ['host_ra', 'host_dec', 'gKronMag', 'rKronMag']
    
    # Create a dummy dataset bank file
    dummy_bank_path = tmp_path / "dummy_bank.csv"
    sample_preprocessed_df.to_csv(dummy_bank_path, index=False)
    
    # Mock build_dataset_bank to verify it's not called when preprocessed_df is provided
    with patch('relaiss.features.build_dataset_bank') as mock_build_dataset, \
         patch('joblib.dump') as mock_dump:
        
        client = rl.ReLAISS()
        client.load_reference()
        client.built_for_AD = True  # Set this flag to use preprocessed_df
        
        # Call train_AD_model with preprocessed_df
        model_path = train_AD_model(
            client=client,
            lc_features=lc_features,
            host_features=host_features,
            preprocessed_df=sample_preprocessed_df,
            path_to_dataset_bank=str(dummy_bank_path),  # Add dummy path
            path_to_models_directory=str(tmp_path),
            force_retrain=True
        )
        
        # Print the call arguments for debugging
        print('joblib.dump call_args_list:', mock_dump.call_args_list)
        
        # Verify build_dataset_bank was not called
        mock_build_dataset.assert_not_called()
        
        # Only the last call should be to save the model data
        last_call_args = mock_dump.call_args_list[-1][0]
        assert 'scaler' in str(last_call_args[0]) or isinstance(last_call_args[0], dict)
        assert last_call_args[1] == model_path
        assert str(tmp_path) in last_call_args[1]
        
        # Check that model_path includes feature counts in the filename
        num_lc_features = len(lc_features)
        num_host_features = len(host_features)
        expected_filename = f"kNN_scaler_lc={num_lc_features}_host={num_host_features}.pkl"
        assert model_path.endswith(expected_filename)

@pytest.mark.skip(reason="Requires real data in CI environment")
def test_train_AD_model_with_raw_data(tmp_path):
    """Test training AD model with raw dataset bank."""
    client = rl.ReLAISS()
    client.load_reference()
    
    model_path = train_AD_model(
        client=client,
        lc_features=client.lc_features,
        host_features=client.host_features,
        path_to_dataset_bank=client.bank_csv,
        path_to_models_directory=str(tmp_path),
        force_retrain=True
    )
    
    assert os.path.exists(model_path)
    model_data = joblib.load(model_path)
    assert 'scaler' in model_data
    assert 'training_features' in model_data

# Updated test that doesn't require real data
def test_train_AD_model_with_mocked_data(tmp_path, sample_preprocessed_df):
    """Test training AD model with a mock dataset bank."""
    # Create a mock dataset bank file
    mock_bank_path = tmp_path / "mock_dataset_bank.csv"
    sample_preprocessed_df.to_csv(mock_bank_path, index=False)
    
    # Define features to use
    lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
    host_features = ['host_ra', 'host_dec', 'gKronMag', 'rKronMag']

    client = rl.ReLAISS()
    client.load_reference()
    client.built_for_AD = False  # Set this flag to use raw data path
    
    # This is overridden later in the test to use the correct kNN naming convention
    
    # Mock the ReLAISS client, build_dataset_bank, and SFDMap to avoid using real dust maps
    with patch('relaiss.relaiss.ReLAISS') as mock_client_class, \
         patch('relaiss.features.build_dataset_bank', return_value=sample_preprocessed_df), \
         patch('sfdmap2.sfdmap.SFDMap') as mock_sfdmap, \
         patch('dust_extinction.parameter_averages.G23') as mock_g23, \
         patch('joblib.dump') as mock_dump:
        
        # Configure mock SFDMap to avoid file access
        mock_map = MagicMock()
        mock_map.ebv.return_value = 0.05  # Mock E(B-V) value
        mock_sfdmap.return_value = mock_map
        
        # Configure mock extinction model
        mock_ext_model = MagicMock()
        mock_ext_model.extinguish.return_value = 0.9  # 10% extinction
        mock_g23.return_value = mock_ext_model
        
        # Configure mock client
        mock_client = MagicMock()
        mock_client.lc_features = lc_features
        mock_client.host_features = host_features
        mock_client.bank_csv = str(mock_bank_path)
        mock_client_class.return_value = mock_client
        
        # Make mock_dump create empty files
        def side_effect(model, path, *args, **kwargs):
            Path(path).touch()
        mock_dump.side_effect = side_effect
        
        # Execute the function
        model_path = train_AD_model(
            client=client,
            lc_features=lc_features,
            host_features=host_features,
            path_to_dataset_bank=str(mock_bank_path),
            path_to_sfd_folder=str(tmp_path),  # Just use tmp_path as mock SFD folder
            path_to_models_directory=str(tmp_path),
            force_retrain=True
        )
        
        # Update expected path for new naming convention
        num_lc_features = len(lc_features)
        num_host_features = len(host_features)
        expected_filename = f"kNN_scaler_lc={num_lc_features}_host={num_host_features}.pkl"
        expected_model_path = str(tmp_path / expected_filename)
        
        # Verify the model path is correct
        assert model_path == expected_model_path
        
        # Verify joblib.dump was called once (only for the model data)
        assert mock_dump.call_count == 1
        
        # The call should be to save the model data dict
        call_args = mock_dump.call_args_list[0][0]
        assert isinstance(call_args[0], dict)
        assert 'scaler' in call_args[0]
        # Check for new optimized format
        assert 'training_sample_scaled' in call_args[0]
        assert 'sample_nbrs' in call_args[0]
        assert 'train_knn_distances' in call_args[0]
        assert 'training_k' in call_args[0]
        assert call_args[1] == model_path

def test_train_AD_model_invalid_input():
    """Test error handling for invalid inputs."""
    client = rl.ReLAISS()
    client.load_reference()
    
    with pytest.raises(ValueError):
        # Neither preprocessed_df nor path_to_dataset_bank provided
        train_AD_model(
            client=client,
            lc_features=['g_peak_mag', 'r_peak_mag'],
            host_features=['host_ra', 'host_dec'],
            preprocessed_df=None,
            path_to_dataset_bank=None
        )

@pytest.fixture
def setup_sfd_data(tmp_path):
    """Setup SFD data directory with dummy files."""
    sfd_dir = tmp_path / "sfd"
    sfd_dir.mkdir()
    for filename in ["SFD_dust_4096_ngp.fits", "SFD_dust_4096_sgp.fits"]:
        (sfd_dir / filename).touch()
    return sfd_dir

@pytest.mark.skip(reason="Requires access to real ZTF data in CI environment")
def test_anomaly_detection_basic(sample_preprocessed_df, tmp_path):
    """Test basic anomaly detection functionality."""
    # Create necessary directories
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    (figure_dir / "AD").mkdir()
    
    # Define features to use
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']

    client = rl.ReLAISS()
    client.load_reference()
    
    # Create a mock model path
    model_path = model_dir / f"kNN_scaler_lc=2_host=2.pkl"
    
    # Create mock timeseries data
    mock_timeseries = pd.DataFrame({
        'mjd': np.linspace(0, 100, 10),
        'mag': np.random.normal(20, 0.5, 10),
        'magerr': np.random.uniform(0.01, 0.1, 10),
        'band': ['g', 'r'] * 5,
        'mjd_cutoff': np.linspace(0, 100, 10),
        'obs_num': range(1, 11),
        'g_peak_mag': [20.0] * 10,
        'r_peak_mag': [19.5] * 10,
        'host_ra': [150.0] * 10, 
        'host_dec': [20.0] * 10
    })
    
    # Mock all the functions that interact with files or external services
    with patch('relaiss.anomaly.train_AD_model', return_value=str(model_path)) as mock_train, \
         patch('relaiss.anomaly.get_timeseries_df', return_value=mock_timeseries) as mock_get_ts, \
         patch('relaiss.anomaly.get_TNS_data', return_value=("TestSN", "Ia", 0.1)), \
         patch('joblib.load'), \
         patch('relaiss.anomaly.check_anom_and_plot'), \
         patch('matplotlib.pyplot.figure', return_value=MagicMock()), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('antares_client.search.get_by_ztf_object_id') as mock_antares:
        
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
        
        # Call anomaly_detection with preprocessed_df
        result = anomaly_detection(
            client=client,
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=str(tmp_path),
            path_to_sfd_folder=str(tmp_path),
            path_to_dataset_bank="dummy_path",  # Should be ignored
            path_to_models_directory=str(model_dir),
            path_to_figure_directory=str(figure_dir),
            preprocessed_df=sample_preprocessed_df  # Use preprocessed_df
        )
        
        # Verify train_AD_model was called with preprocessed_df
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert 'preprocessed_df' in kwargs
        assert kwargs['preprocessed_df'] is sample_preprocessed_df
        
        # Verify get_timeseries_df was called with preprocessed_df
        mock_get_ts.assert_called_once()
        _, kwargs = mock_get_ts.call_args
        assert 'preprocessed_df' in kwargs
        assert kwargs['preprocessed_df'] is sample_preprocessed_df

@pytest.mark.skip(reason="Requires access to real ZTF data in CI environment")
def test_anomaly_detection_with_host_swap(sample_preprocessed_df, tmp_path):
    """Test anomaly detection with host swapping."""
    # Create necessary directories
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    (figure_dir / "AD").mkdir()
    
    # Define features to use
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']

    client = rl.ReLAISS()
    client.load_reference()
    
    # Create a mock model path
    model_path = model_dir / f"kNN_scaler_lc=2_host=2.pkl"
    
    # Create mock timeseries data
    mock_timeseries = pd.DataFrame({
        'mjd': np.linspace(0, 100, 10),
        'mag': np.random.normal(20, 0.5, 10),
        'magerr': np.random.uniform(0.01, 0.1, 10),
        'band': ['g', 'r'] * 5,
        'mjd_cutoff': np.linspace(0, 100, 10),
        'obs_num': range(1, 11),
        'g_peak_mag': [20.0] * 10,
        'r_peak_mag': [19.5] * 10,
        'host_ra': [150.0] * 10, 
        'host_dec': [20.0] * 10
    })
    
    # Mock all the functions that interact with files or external services
    with patch('relaiss.anomaly.train_AD_model', return_value=str(model_path)) as mock_train, \
         patch('relaiss.anomaly.get_timeseries_df', return_value=mock_timeseries) as mock_get_ts, \
         patch('relaiss.anomaly.get_TNS_data', return_value=("TestSN", "Ia", 0.1)), \
         patch('joblib.load'), \
         patch('relaiss.anomaly.check_anom_and_plot'), \
         patch('matplotlib.pyplot.figure', return_value=MagicMock()), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('antares_client.search.get_by_ztf_object_id') as mock_antares:
        
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
        
        # Call anomaly_detection with host swap
        result = anomaly_detection(
            client=client,
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=str(tmp_path),
            path_to_sfd_folder=str(tmp_path),
            path_to_dataset_bank="dummy_path",  # Should be ignored
            path_to_models_directory=str(model_dir),
            path_to_figure_directory=str(figure_dir),
            preprocessed_df=sample_preprocessed_df,  # Use preprocessed_df
            host_ztf_id_to_swap_in="ZTF19aaaaaaa"  # Swap in this host
        )
        
        # Verify train_AD_model was called with preprocessed_df
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert 'preprocessed_df' in kwargs
        assert kwargs['preprocessed_df'] is sample_preprocessed_df
        
        # Verify get_timeseries_df was called with preprocessed_df
        mock_get_ts.assert_called_once()
        _, kwargs = mock_get_ts.call_args
        assert 'preprocessed_df' in kwargs
        assert kwargs['preprocessed_df'] is sample_preprocessed_df
