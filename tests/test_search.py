import pytest
import pandas as pd
import numpy as np
import relaiss as rl

def test_find_neighbors_dataframe():

    client = rl.ReLAISS.load_reference()
    df = client.find_neighbors("ZTF21abbzjeq", k=5, use_host=False)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["ztfid", "distance"]
    assert len(df) == 5
    # Distances should be non-decreasing
    assert np.all(df["distance"].values[:-1] <= df["distance"].values[1:])
