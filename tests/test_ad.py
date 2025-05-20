import pytest
import pandas as pd
import numpy as np
import relaiss as rl
from relaiss.anomaly import *

def test_find_neighbors_dataframe():
    client = rl.ReLAISS()
    client.load_reference(host_features=[])
    df = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=5)

    anomaly_detection(
      transient_ztf_id="ZTF21abbzjeq",
      n_estimators=500,
      contamination=0.02,
      max_samples=10000,
      force_retrain=True
    )

    # if run successful:
    assert True
