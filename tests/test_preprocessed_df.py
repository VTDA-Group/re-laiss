"""Tests for the preprocessed dataframe functionality across the package.

These tests verify that all functions correctly use the preprocessed_df parameter
when provided, bypassing redundant processing operations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import joblib
import os
from pathlib import Path

from relaiss.features import build_dataset_bank, extract_lc_and_host_features
from relaiss.search import primer
from relaiss.fetch import get_timeseries_df
from relaiss.anomaly import train_AD_model, anomaly_detection
import relaiss as rl

@pytest.fixture
def sample_preprocessed_df():
    """Create a sample preprocessed dataframe for testing."""
    np.random.seed(42)
    
    # Create realistic sample data with proper feature ranges
    data = {
        'g_peak_mag': np.random.normal(20, 2, 100),
        'r_peak_mag': np.random.normal(19, 2, 100),
        'g_peak_time': np.random.normal(50, 20, 100),
        'r_peak_time': np.random.normal(55, 20, 100),
        'host_ra': np.random.uniform(0, 360, 100),
        'host_dec': np.random.uniform(-90, 90, 100),
        'zKronMag': np.random.normal(18, 1, 100),
        'wKronMag': np.random.normal(17, 1, 100),
        'jKronMag': np.random.normal(16, 1, 100),
        'hKronMag': np.random.normal(15, 1, 100),
        'ymean': np.random.normal(19, 1, 100),
        'zmean': np.random.normal(18, 1, 100),
        'iminuszKronMag': np.random.normal(0.5, 0.5, 100),
        'ztf_object_id': [f'ZTF{i:08d}' for i in range(100)]
    }
    
    return pd.DataFrame(data)

class TestPreprocessedDataframe:
    """Test suite for preprocessed dataframe functionality across the package."""
    
    def test_build_dataset_bank_with_preprocessed_df(self, tmp_path, sample_preprocessed_df):
        """Verify build_dataset_bank correctly uses a provided preprocessed dataframe."""
        from relaiss.features import build_dataset_bank
        
        # Mock dust map to avoid file access
        with patch('sfdmap2.sfdmap.SFDMap') as mock_sfdmap:
            # Configure mock map
            mock_map = MagicMock()
            mock_map.ebv.return_value = 0.05  # Mock E(B-V) value
            mock_sfdmap.return_value = mock_map
            
            # Call build_dataset_bank with preprocessed_df
            result_df = build_dataset_bank(
                raw_df_bank=None,  # Should be ignored
                preprocessed_df=sample_preprocessed_df,
                path_to_sfd_folder=str(tmp_path)  # Not used but required
            )
            
            # Verify the result matches input
            pd.testing.assert_frame_equal(result_df, sample_preprocessed_df)
    
    def test_primer_with_preprocessed_df(self, tmp_path, sample_preprocessed_df):
        """Verify primer correctly uses a provided preprocessed dataframe."""
        from relaiss.search import primer
        
        # Create a dummy dataset bank file
        dummy_bank_path = tmp_path / "dummy_bank.csv"
        sample_preprocessed_df.to_csv(dummy_bank_path, index=False)
        
        # Use a ZTF ID that exists in the sample preprocessed dataframe
        test_ztf_id = sample_preprocessed_df['ztf_object_id'].iloc[0]
        
        # Call primer with preprocessed_df
        result = primer(
            lc_ztf_id=test_ztf_id,
            theorized_lightcurve_df=None,
            dataset_bank_path=str(dummy_bank_path),
            preprocessed_df=sample_preprocessed_df,
            path_to_timeseries_folder=str(tmp_path),
            path_to_sfd_folder=str(tmp_path)
        )
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'lc_ztf_id' in result
        assert 'locus_feat_arr' in result
    
    def test_get_timeseries_df_with_preprocessed_df(self, tmp_path, sample_preprocessed_df):
        """Verify get_timeseries_df correctly uses a provided preprocessed dataframe."""
        from relaiss.fetch import get_timeseries_df
        
        # Mock ANTARES client and extract_lc_and_host_features to avoid API calls and duplicate columns
        with patch('relaiss.fetch.extract_lc_and_host_features') as mock_extract_lc_host:
            
            # Configure mock to return a proper timeseries dataframe
            mock_extract_lc_host.return_value = pd.DataFrame({
                'ztf_object_id': ['ZTF21abbzjeq'],
                'obs_num': [1],
                'mjd_cutoff': [58000.0],
                'g_peak_mag': [20.0],
                'r_peak_mag': [19.5],
                'g_peak_time': [50.0],
                'r_peak_time': [55.0],
                'host_ra': [150.0],
                'host_dec': [20.0]
            })
            
            # Call get_timeseries_df with preprocessed_df
            result_df = get_timeseries_df(
                ztf_id="ZTF21abbzjeq",
                preprocessed_df=sample_preprocessed_df,
                path_to_timeseries_folder=str(tmp_path),
                path_to_sfd_folder=str(tmp_path)
            )
            
            # Verify the mock was called and returned expected result
            mock_extract_lc_host.assert_called_once()
            
            # Verify result is a dataframe with expected columns
            assert isinstance(result_df, pd.DataFrame)
            assert 'g_peak_mag' in result_df.columns
            assert 'r_peak_mag' in result_df.columns
            assert 'host_ra' in result_df.columns
            assert 'host_dec' in result_df.columns
    
    def test_anomaly_detection_with_preprocessed_df(self, tmp_path, sample_preprocessed_df):
        """Verify anomaly_detection correctly uses a provided preprocessed dataframe."""
        # Create necessary directories
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        figure_dir = tmp_path / "figures"
        figure_dir.mkdir()
        (figure_dir / "AD").mkdir()
        
        # Create proper mock model data with new optimized format
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        mock_scaler = StandardScaler()
        mock_training_data = np.random.random((10, 4))  # 10 samples, 4 features
        mock_scaler.fit(mock_training_data)
        mock_sample_scaled = mock_scaler.transform(mock_training_data[:5])
        mock_sample_nbrs = NearestNeighbors(n_neighbors=5)
        mock_sample_nbrs.fit(mock_sample_scaled)
        
        mock_model_data = {
            'scaler': mock_scaler,
            'training_sample_scaled': mock_sample_scaled,  # Pre-scaled sample
            'sample_nbrs': mock_sample_nbrs,  # Pre-fitted k-NN model
            'train_knn_distances': np.random.random(10) * 2,  # Random distances
            'feature_names': ['g_peak_mag', 'r_peak_mag', 'host_ra', 'host_dec'],
            'training_k': 5
        }
        
        # Mock dependencies
        with patch('relaiss.anomaly.train_AD_model') as mock_train, \
             patch('relaiss.anomaly.get_timeseries_df') as mock_get_ts, \
             patch('relaiss.anomaly.get_TNS_data', return_value=("TestSN", "Ia", 0.1)), \
             patch('joblib.load') as mock_load, \
             patch('relaiss.anomaly.check_anom_and_plot') as mock_check_anom, \
             patch('matplotlib.pyplot.figure', return_value=MagicMock()), \
             patch('matplotlib.pyplot.savefig'), \
             patch('matplotlib.pyplot.close'), \
             patch('relaiss.anomaly.antares_client.search.get_by_ztf_object_id') as mock_antares:
            
            # Configure mock load to return proper model data
            mock_load.return_value = mock_model_data
            
            # Configure mock timeseries with required columns
            mock_ts_df = pd.DataFrame({
                'mjd': np.linspace(0, 100, 10),
                'mag': np.random.normal(20, 0.5, 10),
                'magerr': np.random.uniform(0.01, 0.1, 10),
                'band': ['g', 'r'] * 5,
                'mjd_cutoff': np.linspace(0, 100, 10),  # Add required column
                'g_peak_mag': [20.0] * 10,
                'r_peak_mag': [19.5] * 10,
                'host_ra': [150.0] * 10,
                'host_dec': [20.0] * 10
            })
            mock_get_ts.return_value = mock_ts_df
            
            # Configure mock check_anom_and_plot to return expected tuple
            mock_check_anom.return_value = (50.0, [0.1, 0.2, 0.3], [0.9, 0.8, 0.7])
            
            # Configure mock ANTARES client
            mock_locus = MagicMock()
            mock_ts = MagicMock()
            mock_ts.to_pandas.return_value = pd.DataFrame({
                'ant_mjd': np.linspace(0, 100, 10),
                'ant_passband': ['g', 'R'] * 5,
                'ant_mag': np.random.normal(20, 0.5, 10),
                'ant_magerr': np.random.uniform(0.01, 0.1, 10),
                'ant_ra': [150.0] * 10,
                'ant_dec': [20.0] * 10
            })
            mock_locus.timeseries = mock_ts
            mock_antares.return_value = mock_locus
            
            client = rl.ReLAISS()
            client.load_reference()
            
            # Use a ZTF ID that exists in the sample preprocessed dataframe
            test_ztf_id = sample_preprocessed_df['ztf_object_id'].iloc[0]
            
            # Call anomaly_detection with preprocessed_df
            result = anomaly_detection(
                client=client,
                transient_ztf_id=test_ztf_id,
                lc_features=['g_peak_mag', 'r_peak_mag'],
                host_features=['host_ra', 'host_dec'],
                path_to_timeseries_folder=str(tmp_path),
                path_to_sfd_folder=str(tmp_path),
                path_to_dataset_bank=None,  # Should be ignored
                path_to_models_directory=str(model_dir),
                path_to_figure_directory=str(figure_dir),
                preprocessed_df=sample_preprocessed_df
            )
            
            # Verify train_AD_model was called with preprocessed_df
            mock_train.assert_called_once()
            _, kwargs = mock_train.call_args
            assert kwargs['preprocessed_df'] is sample_preprocessed_df
            
            # Just check that get_timeseries_df was called, and print the call args for debugging
            mock_get_ts.assert_called()
            print('get_timeseries_df call args:', mock_get_ts.call_args)
            
            # Verify check_anom_and_plot was called
            mock_check_anom.assert_called_once()
    
    def test_relaiss_class_with_preprocessed_df(self, tmp_path, sample_preprocessed_df):
        """Verify ReLAISS class methods correctly use a provided preprocessed dataframe."""
        from relaiss.anomaly import anomaly_detection
        
        client = rl.ReLAISS()
        client.load_reference()
        
        # Test that the client can be configured to use preprocessed_df
        client.preprocessed_df = sample_preprocessed_df
        assert client.preprocessed_df is sample_preprocessed_df
        
        # Create proper mock model data with new optimized format
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        mock_scaler = StandardScaler()
        mock_training_data = np.random.random((10, 4))  # 10 samples, 4 features
        mock_scaler.fit(mock_training_data)
        mock_sample_scaled = mock_scaler.transform(mock_training_data[:5])
        mock_sample_nbrs = NearestNeighbors(n_neighbors=5)
        mock_sample_nbrs.fit(mock_sample_scaled)
        
        mock_model_data = {
            'scaler': mock_scaler,
            'training_sample_scaled': mock_sample_scaled,  # Pre-scaled sample
            'sample_nbrs': mock_sample_nbrs,  # Pre-fitted k-NN model
            'train_knn_distances': np.random.random(10) * 2,  # Random distances
            'feature_names': ['g_peak_mag', 'r_peak_mag', 'host_ra', 'host_dec'],
            'training_k': 5
        }
        
        # Mock dependencies to verify they're called with preprocessed_df
        with patch('relaiss.anomaly.train_AD_model') as mock_train, \
             patch('relaiss.search.primer') as mock_primer, \
             patch('relaiss.anomaly.get_timeseries_df') as mock_get_ts, \
             patch('relaiss.anomaly.get_TNS_data', return_value=("TestSN", "Ia", 0.1)), \
             patch('joblib.load', return_value=mock_model_data), \
             patch('relaiss.anomaly.check_anom_and_plot') as mock_check_anom:
            
            # Configure mock timeseries
            mock_ts_df = pd.DataFrame({
                'mjd_cutoff': np.linspace(0, 100, 10),
                'g_peak_mag': [20.0] * 10,
                'r_peak_mag': [19.5] * 10,
                'host_ra': [150.0] * 10,
                'host_dec': [20.0] * 10
            })
            mock_get_ts.return_value = mock_ts_df
            
            # Configure mock check_anom_and_plot
            mock_check_anom.return_value = (50.0, [0.1, 0.2, 0.3], [0.9, 0.8, 0.7])
            
            # Call anomaly_detection with preprocessed_df
            anomaly_detection(
                client=client,
                transient_ztf_id="ZTF21abbzjeq",
                lc_features=['g_peak_mag', 'r_peak_mag'],
                host_features=['host_ra', 'host_dec'],
                path_to_models_directory=str(tmp_path),
                path_to_timeseries_folder=str(tmp_path),
                path_to_sfd_folder=str(tmp_path),
                path_to_dataset_bank=None,
                preprocessed_df=sample_preprocessed_df
            )
            
            # Verify train_AD_model was called with preprocessed_df
            mock_train.assert_called_once()
            _, kwargs = mock_train.call_args
            assert kwargs['preprocessed_df'] is sample_preprocessed_df
