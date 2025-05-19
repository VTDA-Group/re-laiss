import pytest
import pandas as pd
import numpy as np
import relaiss as rl

def test_find_neighbors_dataframe():
    client = rl.ReLAISS()
    client.load_reference(path_to_sfd_folder='/Users/alexgagliano/Documents/Research/ZTF_IInPrecursors/scripts/ztf_forced_phot/sfddata-master/', host_features=[])
    df = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=5)
     # self = Index(['ZTF17aaadkwx', 'ZTF17aabilys', 'ZTF17aabtvsy', 'ZTF17aabvong',
     #  'ZTF17aaaycpc', 'ZTF17aaahhwn', 'ZTF24aaciill',
     #  'ZTF25aajgmfp', 'ZTF24aailqox'],
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert np.all(df["dist"].values[:-1] <= df["dist"].values[1:])
