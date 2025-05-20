import pytest
import pandas as pd
import numpy as np
import relaiss as rl
from relaiss.anomaly import anomaly_detection, train_AD_model
import os
import joblib
from pathlib import Path

@pytest.fixture
def sample_preprocessed_df():
    """Create a sample preprocessed dataframe for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample data with known patterns
    df = pd.DataFrame({
        'g_peak_mag': np.random.normal(20, 1, n_samples),
        'r_peak_mag': np.random.normal(19, 1, n_samples),
        'g_peak_time': np.random.uniform(0, 100, n_samples),
        'r_peak_time': np.random.uniform(0, 100, n_samples),
        'host_ra': np.random.uniform(0, 360, n_samples),
        'host_dec': np.random.uniform(-90, 90, n_samples),
        'gKronMag': np.random.normal(21, 0.5, n_samples),
        'rKronMag': np.random.normal(20, 0.5, n_samples),
        'ztf_object_id': [f'ZTF{i:08d}' for i in range(n_samples)]
    })
    
    # Add some anomalies
    anomaly_idx = np.random.choice(n_samples, size=20, replace=False)
    df.loc[anomaly_idx, 'g_peak_mag'] += 5  # Make these much brighter
    df.loc[anomaly_idx, 'r_peak_mag'] += 5
    
    return df

def test_train_AD_model_with_preprocessed_df(sample_preprocessed_df, tmp_path):
    """Test training AD model with preprocessed dataframe."""
    lc_features = ['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time']
    host_features = ['host_ra', 'host_dec', 'gKronMag', 'rKronMag']
    
    model_path = train_AD_model(
        lc_features=lc_features,
        host_features=host_features,
        preprocessed_df=sample_preprocessed_df,
        path_to_models_directory=str(tmp_path),
        n_estimators=100,
        contamination=0.02,
        max_samples=256,
        force_retrain=True
    )
    
    # Check if model file exists
    assert os.path.exists(model_path)
    
    # Load and verify the model
    model = joblib.load(model_path)
    assert model.n_estimators == 100
    assert model.contamination == 0.02
    assert model.max_samples == 256
    
    # Test model predictions
    X = sample_preprocessed_df[lc_features + host_features].values
    scores = model.predict(X)
    
    # Should find some anomalies (approximately 2% given contamination=0.02)
    n_anomalies = sum(scores == -1)  # IsolationForest uses -1 for anomalies
    expected_anomalies = int(len(X) * 0.02)
    assert abs(n_anomalies - expected_anomalies) < 10  # Allow some variance

def test_train_AD_model_with_raw_data(tmp_path):
    """Test training AD model with raw dataset bank."""
    client = rl.ReLAISS()
    client.load_reference()
    
    model_path = train_AD_model(
        lc_features=client.lc_features,
        host_features=client.host_features,
        path_to_dataset_bank=client.bank_csv,
        path_to_models_directory=str(tmp_path),
        n_estimators=100,
        contamination=0.02,
        max_samples=256,
        force_retrain=True
    )
    
    assert os.path.exists(model_path)
    model = joblib.load(model_path)
    assert model.n_estimators == 100

def test_train_AD_model_invalid_input():
    """Test error handling for invalid inputs."""
    with pytest.raises(ValueError):
        # Neither preprocessed_df nor path_to_dataset_bank provided
        train_AD_model(
            lc_features=['g_peak_mag'],
            host_features=['host_ra'],
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

def test_anomaly_detection_basic(sample_preprocessed_df, tmp_path, setup_sfd_data):
    """Test basic anomaly detection functionality."""
    client = rl.ReLAISS()
    client.load_reference()
    
    # Create necessary directories
    timeseries_dir = tmp_path / "timeseries"
    timeseries_dir.mkdir()
    
    # Run anomaly detection
    anomaly_detection(
        transient_ztf_id="ZTF21abbzjeq",
        lc_features=client.lc_features,
        host_features=client.host_features,
        path_to_timeseries_folder=str(timeseries_dir),
        path_to_sfd_data_folder=str(setup_sfd_data),
        path_to_dataset_bank=client.bank_csv,
        path_to_models_directory=str(tmp_path),
        path_to_figure_directory=str(tmp_path / "figures"),
        save_figures=True,
        n_estimators=100,
        contamination=0.02,
        max_samples=256,
        force_retrain=False
    )
    
    # Check if figures were created
    assert os.path.exists(tmp_path / "figures" / "AD")

def test_anomaly_detection_with_host_swap(sample_preprocessed_df, tmp_path, setup_sfd_data):
    """Test anomaly detection with host galaxy swap."""
    client = rl.ReLAISS()
    client.load_reference()
    
    # Create necessary directories
    timeseries_dir = tmp_path / "timeseries"
    timeseries_dir.mkdir()
    
    # Run anomaly detection with host swap
    anomaly_detection(
        transient_ztf_id="ZTF21abbzjeq",
        lc_features=client.lc_features,
        host_features=client.host_features,
        path_to_timeseries_folder=str(timeseries_dir),
        path_to_sfd_data_folder=str(setup_sfd_data),
        path_to_dataset_bank=client.bank_csv,
        host_ztf_id_to_swap_in="ZTF19aaaaaaa",  # Swap in this host
        path_to_models_directory=str(tmp_path),
        path_to_figure_directory=str(tmp_path / "figures"),
        save_figures=True,
        n_estimators=100,
        contamination=0.02,
        max_samples=256,
        force_retrain=False
    )
    
    # Check if figures were created with host swap suffix
    expected_file = tmp_path / "figures" / "AD" / "ZTF21abbzjeq_w_host_ZTF19aaaaaaa_AD.pdf"
    assert os.path.exists(expected_file)
