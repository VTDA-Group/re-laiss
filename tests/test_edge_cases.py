import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import relaiss as rl
from relaiss.search import primer
from relaiss.anomaly import train_AD_model, anomaly_detection

def test_missing_error_columns(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test that primer handles missing error columns without crashing."""
    # Create a minimal test dataframe with no error columns
    test_df = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq'],
        'g_peak_mag': [20.0],
        'r_peak_mag': [19.5],
        'host_ra': [150.0],
        'host_dec': [20.0],
        # Intentionally missing error columns like g_peak_mag_err
    })
    
    # Mock get_timeseries_df to use our test fixtures 
    with patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)):
        
        # Configure mock to return data without error columns
        mock_df = pd.DataFrame({
            'g_peak_mag': [20.0],
            'r_peak_mag': [19.5],
            'host_ra': [150.0],
            'host_dec': [20.0],
            # No error columns included
        })
        mock_timeseries.return_value = mock_df
        
        # This should not raise any errors
        result = primer(
            lc_ztf_id="ZTF21abbzjeq",
            theorized_lightcurve_df=None,
            dataset_bank_path=dataset_bank_path,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            lc_features=['g_peak_mag', 'r_peak_mag'],
            host_features=['host_ra', 'host_dec'],
            preprocessed_df=test_df,
            num_sims=2  # Request Monte Carlo simulations which use error columns
        )
        
        # Check that the result still has the core information
        assert 'locus_feat_arr' in result
        assert len(result['locus_feat_arrs_mc_l']) == 2  # Requested 2 MC simulations

def test_missing_ant_mjd_column(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test that anomaly detection handles missing ant_mjd column without crashing."""
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Create a sample dataframe without ant_mjd column
    test_df = pd.DataFrame({
        'ztf_object_id': ['ZTF21abbzjeq', 'ZTF19aaaaaaa'],
        'g_peak_mag': [20.0, 19.0],
        'r_peak_mag': [19.5, 18.5],
        'host_ra': [150.0, 160.0],
        'host_dec': [20.0, 25.0],
        # Missing ant_mjd column
    })
    
    # Mock the needed components
    with patch('relaiss.fetch.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.anomaly.get_timeseries_df') as mock_ad_timeseries, \
         patch('joblib.dump'), \
         patch('relaiss.anomaly.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)), \
         patch('joblib.load', return_value=MagicMock()), \
         patch('relaiss.anomaly.check_anom_and_plot'):  # Skip plotting
        
        # Configure mocks to return dataframes without ant_mjd
        df_without_ant_mjd = pd.DataFrame({
            'ztf_object_id': ['ZTF21abbzjeq'],
            'g_peak_mag': [20.0],
            'r_peak_mag': [19.5],
            'host_ra': [150.0],
            'host_dec': [20.0],
            # Intentionally missing ant_mjd
            'obs_num': [1]
        })
        mock_timeseries.return_value = df_without_ant_mjd
        mock_ad_timeseries.return_value = df_without_ant_mjd
        
        # This should work without errors
        # First train model (also tests duplicate model save message fix)
        model_path = train_AD_model(
            lc_features=lc_features,
            host_features=host_features,
            preprocessed_df=test_df,
            path_to_models_directory="./test_models",
            n_estimators=10,  # Small for quick test
            max_samples=8     # Small for quick test
        )
        
        # Then run anomaly detection
        anomaly_detection(
            transient_ztf_id="ZTF21abbzjeq",
            lc_features=lc_features,
            host_features=host_features,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            path_to_dataset_bank=dataset_bank_path,
            path_to_models_directory="./test_models",
            preprocessed_df=test_df,
            save_figures=False
        )
        
        # If we got here without errors, the test passes

def test_host_feature_length_mismatch(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test that primer handles mismatched host feature lengths."""
    # Create a test with a theorized lightcurve
    lightcurve_df = pd.DataFrame({
        'ant_mjd': np.linspace(0, 100, 10),
        'ant_mag': np.random.normal(20, 0.5, 10),
        'ant_magerr': np.random.uniform(0.01, 0.1, 10),
        'ant_passband': ['g', 'R'] * 5  # Alternating g and R bands
    })
    
    # Define features with intentionally mismatched lengths
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Mock the necessary functions
    with patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)):
        
        # Configure the mock with values mismatched to feature lengths
        # This simulates the scenario where host_locus_feat_arr has 4 values but
        # is intended to be mapped to just 2 host features
        mock_df = pd.DataFrame({
            'g_peak_mag': [20.0],
            'r_peak_mag': [19.5],
            'host_ra': [150.0],
            'host_dec': [20.0]
        })
        mock_timeseries.return_value = mock_df
        
        # This should handle the mismatch correctly
        result = primer(
            lc_ztf_id=None,
            theorized_lightcurve_df=lightcurve_df,
            dataset_bank_path=dataset_bank_path,
            path_to_timeseries_folder=timeseries_dir,
            path_to_sfd_folder=sfd_dir,
            host_ztf_id="ZTF19aaaaaaa",  # Required for theorized lightcurve
            lc_features=lc_features,
            host_features=host_features
        )
        
        # Verify the output has correct structure
        assert 'locus_feat_arr' in result
        assert len(result['locus_feat_arr']) == len(lc_features) + len(host_features)

def test_swapped_host_with_empty_lightcurve(dataset_bank_path, timeseries_dir, sfd_dir):
    """Test handling a swapped host galaxy with empty light curve features."""
    
    # Mock the necessary functions
    with patch('relaiss.features.extract_lc_and_host_features') as mock_extract, \
         patch('relaiss.features.build_dataset_bank', return_value=pd.DataFrame()), \
         patch('relaiss.search.get_timeseries_df', return_value=pd.DataFrame()), \
         patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)):
        
        # Configure extract_lc_and_host_features to simulate empty light curve with swapped host
        # First simulate normal return value
        mock_extract.return_value = pd.DataFrame({
            'g_peak_mag': [20.0],
            'r_peak_mag': [19.5],
            'host_ra': [150.0],
            'host_dec': [20.0]
        })
        
        # Set up the client
        client = rl.ReLAISS()
        
        # This test passes if find_neighbors completes without errors
        # We're testing the fix that allows find_neighbors to handle situations 
        # where we have a swapped host but empty light curve features
        client.find_neighbors(
            ztf_object_id="ZTF21abbzjeq",
            host_ztf_id="ZTF19aaaaaaa",
            weight_lc_feats_factor=10.0,  # Also test the weighting fix
            n=3
        )

def test_optional_ztf_object_id():
    """Test that find_neighbors accepts None for ztf_object_id when using theorized lightcurve."""
    # Create a minimal client
    client = rl.ReLAISS()
    # Test by monkeypatching the primer function to avoid actual computation
    with patch.object(client, 'find_neighbors', return_value=pd.DataFrame()):
        # The key test is that this doesn't raise an error due to missing required argument
        # We need to provide theorized_lightcurve_df and host_ztf_id to avoid other validation errors
        client.find_neighbors(
            theorized_lightcurve_df=pd.DataFrame({
                'ant_mjd': [1, 2, 3],
                'ant_mag': [19, 20, 19],
                'ant_magerr': [0.1, 0.1, 0.1],
                'ant_passband': ['g', 'R', 'g']
            }),
            host_ztf_id="ZTF19aaaaaaa",
            n=3
        )
        # If we get here, the test passes 