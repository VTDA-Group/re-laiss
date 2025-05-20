import pytest
import pandas as pd
import numpy as np
import relaiss as rl

def test_find_neighbors_dataframe():
    client = rl.ReLAISS()
    client.load_reference(host_features=[])
    df = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=5, plot=True, save_figures=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert np.all(df["dist"].values[:-1] <= df["dist"].values[1:])
