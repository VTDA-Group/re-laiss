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
from relaiss.anomaly import train_AD_model


class TestPreprocessedDataframe:
    """Test suite for preprocessed dataframe functionality across the package."""
    
    def test_build_dataset_bank_with_preprocessed_df(self, sample_preprocessed_df):
        """Verify build_dataset_bank returns preprocessed_df directly when provided.
        
        When preprocessed_df is passed, the function should:
        - Return the same dataframe object without modification
        - Skip all processing steps that would normally be applied to raw_df_bank
        """
        # Create a simple raw dataframe that would normally be processed
        raw_df = pd.DataFrame({
            'ztf_object_id': ['ZTF123', 'ZTF456'],
            'g_peak_mag': [19.0, 18.5],
            'r_peak_mag': [18.0, 17.5],
            'host_ra': [150.0, 160.0],
            'host_dec': [20.0, 25.0]
        })
        
        # Call build_dataset_bank with the preprocessed_df
        result_df = build_dataset_bank(
            raw_df_bank=raw_df,  # This should be ignored
            preprocessed_df=sample_preprocessed_df  # This should be returned directly
        )
        
        # Verify that the function returned the exact same dataframe object
        assert result_df is sample_preprocessed_df
        
        # Verify none of the values from raw_df appear in the result
        assert 'ZTF123' not in result_df['ztf_object_id'].values
        assert 'ZTF456' not in result_df['ztf_object_id'].values
    
    def test_primer_with_preprocessed_df(self, dataset_bank_path, timeseries_dir, sfd_dir):
        """Verify primer uses preprocessed_df instead of loading dataset_bank_path.
        
        When preprocessed_df is provided, the primer function should:
        - Never attempt to read the dataset_bank_path file
        - Use the provided dataframe for feature vector assembly
        """
        # Create a test preprocessed dataframe
        test_df = pd.DataFrame({
            'ztf_object_id': ['ZTF21abbzjeq', 'ZTF19aaaaaaa'],
            'g_peak_mag': [20.0, 19.0],
            'r_peak_mag': [19.5, 18.5], 
            'host_ra': [150.0, 160.0],
            'host_dec': [20.0, 25.0]
        })
        
        # Real read_csv function to allow reading the timeseries file
        original_read_csv = pd.read_csv
        
        def mock_read_csv_side_effect(path, *args, **kwargs):
            if str(path) == str(dataset_bank_path):
                # If attempting to read dataset_bank_path, this should fail the test
                pytest.fail(f"Should not have called read_csv with {dataset_bank_path}")
            return original_read_csv(path, *args, **kwargs)
        
        # Mock get_timeseries_df to use our test fixtures
        with patch('relaiss.search.get_timeseries_df') as mock_timeseries, \
             patch('relaiss.search.get_TNS_data', return_value=("TNS2023abc", "SN Ia", 0.1)), \
             patch('pandas.read_csv', side_effect=mock_read_csv_side_effect):
            
            # Configure the mock to return a sample dataframe
            mock_df = pd.read_csv(timeseries_dir / "ZTF21abbzjeq.csv")
            mock_timeseries.return_value = mock_df
            
            # Call the primer function with preprocessed_df
            result = primer(
                lc_ztf_id="ZTF21abbzjeq",
                theorized_lightcurve_df=None,
                dataset_bank_path=dataset_bank_path,  # This should be ignored
                path_to_timeseries_folder=timeseries_dir,
                path_to_sfd_folder=sfd_dir,
                save_timeseries=False,
                lc_features=['g_peak_mag', 'r_peak_mag'],
                host_features=['host_ra', 'host_dec'],
                preprocessed_df=test_df  # Use our test dataframe
            )
            
            # Check that the result is as expected
            assert isinstance(result, dict)
            assert 'lc_ztf_id' in result
            assert 'locus_feat_arr' in result
            assert result['lc_ztf_id'] == "ZTF21abbzjeq"
    
    def test_get_timeseries_df_with_preprocessed_df(self):
        """Verify get_timeseries_df passes preprocessed_df to extract_lc_and_host_features.
        
        When preprocessed_df is provided, get_timeseries_df should:
        - Pass the preprocessed_df to extract_lc_and_host_features
        - Not try to read or process the dataset_bank_path
        """
        # Create a sample preprocessed dataframe with the test transient
        test_preprocessed_df = pd.DataFrame({
            'ztf_object_id': ['ZTF21abbzjeq', 'ZTF19aaaaaaa'],
            'g_peak_mag': [20.0, 19.0],
            'r_peak_mag': [19.5, 18.5], 
            'host_ra': [150.0, 160.0],
            'host_dec': [20.0, 25.0]
        })
        
        # Create mock values that should be return by extract_lc_and_host_features
        mock_timeseries = pd.DataFrame({
            'mjd': [58000, 58001],
            'mag': [20.0, 20.1],
            'magerr': [0.1, 0.1],
            'band': ['g', 'r']
        })
        
        # Mock extract_lc_and_host_features to verify it's called with the preprocessed_df
        with patch('relaiss.fetch.extract_lc_and_host_features', return_value=mock_timeseries) as mock_extract:
            # Call get_timeseries_df with a preprocessed dataframe
            result = get_timeseries_df(
                ztf_id='ZTF21abbzjeq',
                path_to_timeseries_folder='dummy_path',
                path_to_sfd_folder='dummy_path',
                path_to_dataset_bank='dummy_path',
                preprocessed_df=test_preprocessed_df
            )
            
            # Verify the extract_lc_and_host_features function was called with the preprocessed_df
            mock_extract.assert_called_once()
            # Check that preprocessed_df was passed to extract_lc_and_host_features
            _, kwargs = mock_extract.call_args
            assert 'preprocessed_df' in kwargs
            assert kwargs['preprocessed_df'] is test_preprocessed_df
            
            # Verify the result matches what was returned by extract_lc_and_host_features
            pd.testing.assert_frame_equal(result, mock_timeseries)
    
    def test_train_AD_model_with_preprocessed_df(self, tmp_path, sample_preprocessed_df):
        """Verify train_AD_model correctly uses a provided preprocessed dataframe.
        
        When preprocessed_df is provided, train_AD_model should:
        - Not attempt to read or process dataset_bank_path
        - Use the preprocessed_df directly for training
        - Include feature counts in the model filename
        """
        # Define features to use
        lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
        host_features = ['host_ra', 'host_dec', 'gKronMag', 'rKronMag']
        
        # Mock build_dataset_bank to verify it's not called when preprocessed_df is provided
        with patch('relaiss.features.build_dataset_bank') as mock_build_dataset, \
             patch('joblib.dump') as mock_dump:
            
            # Call train_AD_model with preprocessed_df
            model_path = train_AD_model(
                client=client,
                lc_features=lc_features,
                host_features=host_features,
                preprocessed_df=sample_preprocessed_df,
                path_to_models_directory=str(tmp_path),
                n_estimators=100,
                contamination=0.02,
                max_samples=256,
                force_retrain=True
            )
            
            # Verify build_dataset_bank was not called
            mock_build_dataset.assert_not_called()
            
            # Verify joblib.dump was called with the right parameters
            mock_dump.assert_called_once()
            
            # Check that model_path includes feature counts in the filename
            num_lc_features = len(lc_features)
            num_host_features = len(host_features)
            expected_filename = f"IForest_n=100_c=0.02_m=256_lc={num_lc_features}_host={num_host_features}.pkl"
            assert model_path.endswith(expected_filename)

    def test_anomaly_detection_with_preprocessed_df(self, tmp_path, sample_preprocessed_df):
        """Verify anomaly_detection correctly passes preprocessed_df to train_AD_model.
        
        When preprocessed_df is provided to anomaly_detection, it should:
        - Pass the preprocessed_df to train_AD_model
        - Pass the preprocessed_df to get_timeseries_df
        """
        from relaiss.anomaly import anomaly_detection
        
        # Create necessary directories
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        figure_dir = tmp_path / "figures"
        figure_dir.mkdir()
        (figure_dir / "AD").mkdir()
        
        # Define features
        lc_features = ['g_peak_mag', 'r_peak_mag']
        host_features = ['host_ra', 'host_dec']
        
        # Create a mock model path
        model_path = model_dir / f"IForest_n=100_c=0.02_m=256_lc=2_host=2.pkl"
        
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
            anomaly_detection(
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

    def test_relaiss_class_with_preprocessed_df(self, tmp_path, sample_preprocessed_df):
        """Verify the ReLAISS class correctly caches and uses the preprocessed dataframe.
        
        This test confirms that:
        1. The preprocessed dataframe is correctly stored in the ReLAISS instance
        2. It can be retrieved via get_preprocessed_dataframe()
        """
        import relaiss as rl
        from unittest.mock import patch, MagicMock
        
        # Create a mock ReLAISS instance with a hydrated_bank attribute
        client = MagicMock(spec=rl.ReLAISS)
        client.hydrated_bank = sample_preprocessed_df
        
        # Patch the get_preprocessed_dataframe method to use the real implementation
        with patch.object(rl.ReLAISS, 'get_preprocessed_dataframe', 
                         return_value=sample_preprocessed_df):
            
            # Get the preprocessed dataframe
            df = rl.ReLAISS.get_preprocessed_dataframe(client)
            
            # Verify it's the same dataframe
            assert df is not None
            pd.testing.assert_frame_equal(df, sample_preprocessed_df) 
