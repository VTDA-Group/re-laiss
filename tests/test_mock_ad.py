import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
from relaiss.anomaly import train_AD_model
from pyod.models.iforest import IForest
from relaiss.relaiss import ReLAISS
import pickle

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

    client = ReLAISS()
    client.load_reference()
    client.built_for_AD = True  # Set this flag to use preprocessed_df
    
    # Create a mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.fit.return_value = mock_pipeline
    
    # Create a temporary directory for models
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('relaiss.anomaly.Pipeline', return_value=mock_pipeline), \
             patch('joblib.dump') as mock_dump:
            
            # Make mock_dump actually create an empty file
            def side_effect(model, path, *args, **kwargs):
                Path(path).touch()
            mock_dump.side_effect = side_effect
            
            # Train model
            model_path = train_AD_model(
                client=client,
                lc_features=lc_features,
                host_features=host_features,
                preprocessed_df=df,
                path_to_models_directory=tmpdir,
                n_estimators=100,
                contamination=0.02,
                max_samples=256,
                force_retrain=True
            )
            
            # Check if model file exists and has correct feature counts in name
            assert os.path.exists(model_path)
            assert "IForest" in os.path.basename(model_path)
            assert "n=100" in os.path.basename(model_path)
            assert f"lc={len(lc_features)}" in os.path.basename(model_path)
            assert f"host={len(host_features)}" in os.path.basename(model_path)
            
            # Verify joblib.dump was called with our mock pipeline
            mock_dump.assert_called_once()
            saved_pipeline = mock_dump.call_args[0][0]
            assert saved_pipeline is mock_pipeline

def test_train_AD_model_error_handling():
    """Test error handling in train_AD_model."""
    client = ReLAISS()
    client.load_reference()
    
    # Test with missing required parameters
    with pytest.raises(ValueError):
        train_AD_model(
            client=client,
            lc_features=['g_peak_mag'],
            host_features=['host_ra'],
            preprocessed_df=None,
            path_to_dataset_bank=None
        )

def test_train_AD_model_with_joblib():
    """Test train_AD_model with joblib mocked."""
    # Create sample data
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
    })

    client = ReLAISS()
    client.load_reference()
    client.built_for_AD = True  # Set this flag to use preprocessed_df
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('relaiss.anomaly.IForest') as mock_iforest_class:
            # Create mock model with all required attributes and methods
            mock_model = MagicMock(spec=IForest)
            mock_model.n_estimators = 500
            mock_model.contamination = 0.02
            mock_model.max_samples = 1024
            mock_model.fit.return_value = mock_model
            mock_model.decision_function.return_value = np.array([0.5, 0.6, 0.7])
            mock_model.predict.return_value = np.array([0, 0, 1])
            mock_iforest_class.return_value = mock_model
            
            with patch('joblib.dump') as mock_dump:
                # Make mock_dump actually create an empty file
                def side_effect(model, path, *args, **kwargs):
                    Path(path).touch()
                mock_dump.side_effect = side_effect
                
                # Call train_AD_model
                model_path = train_AD_model(
                    client=client,
                    lc_features=['feature1'],
                    host_features=['feature2'],
                    preprocessed_df=df,
                    path_to_models_directory=tmpdir,
                    force_retrain=True
                )
                
                # Verify that the Isolation Forest constructor was called with expected parameters
                mock_iforest_class.assert_called_once()
                _, kwargs = mock_iforest_class.call_args
                assert kwargs['n_estimators'] == 500
                assert kwargs['contamination'] == 0.02
                assert kwargs['max_samples'] == 1024
                
                # Check that joblib.dump was called with the right arguments
                mock_dump.assert_called_once()
                saved_pipeline = mock_dump.call_args[0][0]
                # Verify it's a Pipeline object
                assert 'Pipeline' in str(type(saved_pipeline))
                # Verify our mock model is in the pipeline
                assert any(step[1] is mock_model for step in saved_pipeline.steps)

def test_anomaly_detection_simplified(tmp_path):
    """Test anomaly detection with minimal dependencies."""
    from relaiss.anomaly import anomaly_detection
    
    # Create the necessary directories
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir(exist_ok=True)
    (figure_dir / "AD").mkdir(exist_ok=True)
    
    # Create a real IForest that can be pickled
    real_forest = IForest(n_estimators=10, random_state=42)
    X = np.random.rand(20, 4)  # Some dummy data
    real_forest.fit(X)  # Fit the model so it can be used
    
    # Create a dummy model file with actual content
    model_path = model_dir / "IForest_n=100_c=0.02_m=256_lc=2_host=2.pkl"  # Use actual filename format with feature counts
    
    # Write the real forest to the file
    with open(model_path, 'wb') as f:
        pickle.dump(real_forest, f)
    
    # Features to test with - should match the dimensions used to create X above
    lc_features = ['g_peak_mag', 'r_peak_mag']
    host_features = ['host_ra', 'host_dec']
    
    # Create preprocessed dataframe
    df = pd.DataFrame({
        'g_peak_mag': np.random.normal(20, 1, 100),
        'r_peak_mag': np.random.normal(19, 1, 100),
        'host_ra': np.random.uniform(0, 360, 100),
        'host_dec': np.random.uniform(-90, 90, 100),
    })
    
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
        
        # Mock check_anom_and_plot to return proper values
        mock_check_anom.return_value = (np.array([58000.0]), np.array([0.5]), np.array([0.75]))

        client = ReLAISS()
        client.load_reference()
        client.built_for_AD = True  # Set this flag to use preprocessed_df
        
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
            force_retrain=False,
            preprocessed_df=df,  # Add preprocessed_df parameter
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
