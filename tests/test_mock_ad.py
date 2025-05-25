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

def test_anomaly_detection_simplified():
    """Test anomaly detection with minimal dependencies."""
    from relaiss.anomaly import anomaly_detection
    
    # Create necessary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "models"
        model_dir.mkdir(exist_ok=True)
        figure_dir = Path(tmpdir) / "figures"
        figure_dir.mkdir(exist_ok=True)
        (figure_dir / "AD").mkdir(exist_ok=True)
        
        # Create IForest model
        real_forest = IForest(n_estimators=10, random_state=42)
        X = np.random.rand(20, 4)
        real_forest.fit(X)
        
        # Define features first
        lc_features = ['g_peak_mag', 'r_peak_mag']
        host_features = ['host_ra', 'host_dec']
        
        client = ReLAISS()
        client.load_reference()
        client.built_for_AD = True  # Set this flag to use preprocessed_df
        
        model_path = model_dir / "IForest_n=100_c=0.02_m=256.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(real_forest, f)
        
        # Create preprocessed dataframe
        df = pd.DataFrame({
            'g_peak_mag': np.random.normal(20, 1, 100),
            'r_peak_mag': np.random.normal(19, 1, 100),
            'host_ra': np.random.uniform(0, 360, 100),
            'host_dec': np.random.uniform(-90, 90, 100),
        })
