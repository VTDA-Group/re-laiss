import pytest
import pandas as pd
import numpy as np

import relaiss as rl


@pytest.fixture(scope="session")
def relaiss_client():
    """Load the cached reference client once for the whole test session."""
    try:
        client = rl.load_reference()
    except FileNotFoundError as err:
        pytest.skip(f"Reference index unavailable – {err}")
    return client


def test_load_reference_singleton(relaiss_client):
    c1 = rl.load_reference()
    c2 = rl.load_reference()
    assert c1 is c2, "load_reference should cache the client instance"


def test_find_neighbors_dataframe(relaiss_client):
    df = rl.find_neighbors("ZTF21abbzjeq", k=5)  # arbitrary real ZTF ID
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["ztfid", "distance"]
    assert len(df) == 5
    # Distances should be non-decreasing
    assert np.all(df["distance"].values[:-1] <= df["distance"].values[1:])
