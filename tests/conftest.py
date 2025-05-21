import pytest
import os
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import astropy.units as u
from .fixtures.search import build_test_annoy_index

# Get the path to the test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURE_DATA_DIR = FIXTURES_DIR / "data"
FIXTURE_SFD_DIR = FIXTURES_DIR / "sfd"
FIXTURE_TIMESERIES_DIR = FIXTURES_DIR / "timeseries"

@pytest.fixture
def fixture_dir():
    """Return the path to the test fixtures directory."""
    return FIXTURES_DIR

@pytest.fixture
def data_dir():
    """Return the path to the test data directory."""
    return FIXTURE_DATA_DIR

@pytest.fixture
def sfd_dir():
    """Return the path to the test SFD directory."""
    return FIXTURE_SFD_DIR

@pytest.fixture
def timeseries_dir():
    """Return the path to the test timeseries directory."""
    return FIXTURE_TIMESERIES_DIR

@pytest.fixture
def dataset_bank_path():
    """Return the path to the test dataset bank CSV file."""
    return FIXTURE_DATA_DIR / "dataset_bank.csv"

@pytest.fixture
def test_dataframe():
    """Create a test dataframe with random data."""
    n_samples = 100
    np.random.seed(42)
    
    return pd.DataFrame({
        'g_peak_mag': np.random.normal(20, 1, n_samples),
        'r_peak_mag': np.random.normal(19, 1, n_samples),
        'g_peak_time': np.random.uniform(0, 100, n_samples),
        'r_peak_time': np.random.uniform(0, 100, n_samples),
        'host_ra': np.random.uniform(0, 360, n_samples),
        'host_dec': np.random.uniform(-90, 90, n_samples),
        'ztf_object_id': [f'ZTF{i:08d}' for i in range(n_samples)]
    })

@pytest.fixture
def mock_antares_client():
    """Mock the ANTARES client for testing."""
    with patch('antares_client.search.get_by_ztf_object_id') as mock_get:
        # Create a mock locus object
        mock_locus = MagicMock()
        
        # Set up the mock timeseries
        mock_ts = MagicMock()
        mock_df = pd.DataFrame({
            'ant_mjd': np.linspace(0, 100, 50),
            'ant_passband': ['g', 'r'] * 25,
            'ant_mag': np.random.normal(20, 0.5, 50),
            'ant_magerr': np.random.uniform(0.01, 0.1, 50)
        })
        mock_ts.to_pandas.return_value = mock_df
        mock_locus.timeseries = mock_ts
        
        # Set up catalog objects for TNS data
        mock_locus.catalog_objects = {
            "tns_public_objects": [
                {"name": "TNS2023abc", "type": "SN Ia", "redshift": 0.1}
            ]
        }
        
        # Configure the mock function to return our mock locus
        mock_get.return_value = mock_locus
        
        yield mock_get

@pytest.fixture
def mock_tns_data():
    """Mock the TNS data retrieval function."""
    with patch('relaiss.fetch.get_TNS_data') as mock_tns:
        mock_tns.return_value = ("TNS2023abc", "SN Ia", 0.1)
        yield mock_tns

@pytest.fixture
def mock_ps1_cutout():
    """Mock the PS1 cutout function."""
    with patch('relaiss.fetch.fetch_ps1_cutout') as mock_fetch:
        # Create a mock array for testing
        mock_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mock_fetch.return_value = mock_array
        yield mock_fetch

@pytest.fixture
def mock_ps1_rgb_jpeg():
    """Mock the PS1 RGB JPEG function."""
    with patch('relaiss.fetch.fetch_ps1_rgb_jpeg') as mock_fetch:
        # Create a mock RGB array for testing
        mock_rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_fetch.return_value = mock_rgb
        yield mock_fetch

@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib to avoid creating actual plots during tests."""
    with patch('matplotlib.pyplot.figure', return_value=MagicMock()), \
         patch('matplotlib.pyplot.savefig'), \
         patch('matplotlib.pyplot.close'), \
         patch('matplotlib.pyplot.show'):
        yield

@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for saving models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def temp_figure_dir():
    """Create a temporary directory for saving figures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

# Fixture to mock dust extinction modules
@pytest.fixture
def mock_sfdmap():
    """Mock the SFDMap class."""
    with patch('sfdmap2.sfdmap.SFDMap') as mock_sfd:
        # Create a mock SFDMap instance
        mock_map = MagicMock()
        mock_map.ebv.return_value = 0.05  # Mock E(B-V) value
        mock_sfd.return_value = mock_map
        yield mock_sfd

# Fixture to mock dust extinction calculation
@pytest.fixture
def mock_extinction():
    """Mock the dust extinction modules."""
    # Create a mock extinction model
    ext_model = MagicMock()
    
    # Handle different versions of the extinguish method
    def mock_extinguish(*args, **kwargs):
        # Just return a fixed attenuation value regardless of arguments
        if 'Ebv' in kwargs or 'Av' in kwargs or len(args) > 1:
            # If used with wavelength array, return array of same shape
            if len(args) > 0 and hasattr(args[0], 'shape'):
                return np.ones_like(args[0]) * 0.9  # 10% extinction
            return 0.9  # 10% extinction
        return 0.9  # Default extinction value
    
    ext_model.extinguish = mock_extinguish
    
    # Create the actual patches
    with patch('dust_extinction.parameter_averages.G23', return_value=ext_model), \
         patch('sfdmap2.sfdmap.SFDMap') as mock_sfdmap:
        
        # Setup the mock SFD map
        mock_map = MagicMock()
        mock_map.ebv.return_value = 0.05  # Mock E(B-V) value
        mock_sfdmap.return_value = mock_map
        
        yield

@pytest.fixture
def mock_extinction_all():
    """Mock all extinction-related functions."""
    with patch('sfdmap2.sfdmap.SFDMap') as mock_map, \
         patch('dust_extinction.parameter_averages.G23') as mock_g23, \
         patch('astropy.units.um', u.um):
        
        # Configure mocks
        mock_sfd = MagicMock()
        mock_sfd.ebv.return_value = 0.05
        mock_map.return_value = mock_sfd
        
        mock_ext_model = MagicMock()
        mock_ext_model.extinguish.return_value = 0.9  # 10% extinction
        mock_g23.return_value = mock_ext_model
        
        yield mock_map, mock_g23

@pytest.fixture
def test_annoy_index(dataset_bank_path):
    """Create a test Annoy index for testing."""
    # Get the dataset bank file
    df = pd.read_csv(dataset_bank_path)
    
    # Build a temporary Annoy index
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "annoy_index"
        index, index_path, object_ids = build_test_annoy_index(
            test_databank_path=dataset_bank_path,
            lc_features=['g_peak_mag', 'r_peak_mag', 'g_peak_time', 'r_peak_time'],
            host_features=['host_ra', 'host_dec']
        )
        yield index, index_path, object_ids

@pytest.fixture
def mock_timeseries():
    """Mock the timeseries data retrieval."""
    with patch('relaiss.fetch.get_timeseries_df') as mock_ts:
        # Create a test timeseries dataframe
        df = pd.read_csv(FIXTURE_TIMESERIES_DIR / "ZTF21abbzjeq.csv")
        mock_ts.return_value = df
        yield mock_ts

@pytest.fixture
def sample_preprocessed_df():
    """Create a sample preprocessed dataframe for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create sample data with required columns
    df = pd.DataFrame({
        # Lightcurve features
        'g_peak_mag': np.random.normal(20, 1, n_samples),
        'r_peak_mag': np.random.normal(19, 1, n_samples),
        'g_peak_time': np.random.uniform(0, 100, n_samples),
        'r_peak_time': np.random.uniform(0, 100, n_samples),
        # Host galaxy position
        'ra': np.random.uniform(0, 360, n_samples),
        'dec': np.random.uniform(-90, 90, n_samples),
        'host_ra': np.random.uniform(0, 360, n_samples),
        'host_dec': np.random.uniform(-90, 90, n_samples),
        # g-band Kron measurements
        'gKronMag': np.random.normal(21, 0.5, n_samples),
        'gKronMagErr': np.random.uniform(0.01, 0.2, n_samples),
        'gKronRad': np.random.uniform(1, 5, n_samples),
        'gExtNSigma': np.random.uniform(1, 3, n_samples),
        # r-band Kron measurements
        'rKronMag': np.random.normal(20, 0.5, n_samples),
        'rKronMagErr': np.random.uniform(0.01, 0.2, n_samples),
        'rKronRad': np.random.uniform(1, 5, n_samples),
        'rExtNSigma': np.random.uniform(1, 3, n_samples),
        # i-band Kron measurements
        'iKronMag': np.random.normal(19.5, 0.5, n_samples),
        'iKronMagErr': np.random.uniform(0.01, 0.2, n_samples),
        'iKronRad': np.random.uniform(1, 5, n_samples),
        'iExtNSigma': np.random.uniform(1, 3, n_samples),
        # z-band Kron measurements
        'zKronMag': np.random.normal(19, 0.5, n_samples),
        'zKronMagErr': np.random.uniform(0.01, 0.2, n_samples),
        'zKronRad': np.random.uniform(1, 5, n_samples),
        'zExtNSigma': np.random.uniform(1, 3, n_samples),
        # Moment information
        'rmomentXX': np.random.uniform(0.5, 2, n_samples),
        'rmomentYY': np.random.uniform(0.5, 2, n_samples),
        'rmomentXY': np.random.uniform(-0.5, 0.5, n_samples),
        # ZTF IDs
        'ztf_object_id': [f'ZTF{i:08d}' for i in range(n_samples)]
    })
    
    # Add corrected magnitudes and color features that would normally be engineered
    df['gKronMagCorrected'] = df['gKronMag'] - 0.1  # Mock extinction correction
    df['rKronMagCorrected'] = df['rKronMag'] - 0.08
    df['iKronMagCorrected'] = df['iKronMag'] - 0.06
    df['zKronMagCorrected'] = df['zKronMag'] - 0.04
    
    # Add color features
    df['gminusrKronMag'] = df['gKronMag'] - df['rKronMag']
    df['rminusiKronMag'] = df['rKronMag'] - df['iKronMag']
    df['iminuszKronMag'] = df['iKronMag'] - df['zKronMag']
    
    return df

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "relaiss: mark a test as requiring the full relaiss environment"
    )

def pytest_collection_modifyitems(config, items):
    """Skip tests that require the full relaiss environment in CI."""
    # Check if we're in a CI environment
    ci_mode = config.getoption("--ci", default=False)
    if ci_mode:
        skip_relaiss = pytest.mark.skip(reason="Requires full relaiss environment with external data")
        for item in items:
            if "relaiss" in item.keywords:
                item.add_marker(skip_relaiss)
            
            # Skip any test that accesses real data paths by default
            if any(pattern in str(item.function) for pattern in [
                "../data/",
                "ZTF21abbzjeq",
                "find_neighbors",
                "load_reference",
                "path_to_sfd_folder",
                "path_to_dataset_bank",
                "load_cached_dataframe",
                "path_to_timeseries_folder",
            ]):
                item.add_marker(skip_relaiss)

def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--ci", action="store_true", default=False, help="Run in CI mode (skip tests requiring real data)"
    ) 