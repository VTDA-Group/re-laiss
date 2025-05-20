import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import relaiss as rl
from relaiss.plotting import plot_lightcurves, plot_hosts, corner_plot

def test_find_neighbors_dataframe():
    client = rl.ReLAISS()
    client.load_reference(host_features=[])
    df = client.find_neighbors(ztf_object_id="ZTF21abbzjeq", n=5, plot=True, save_figures=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert np.all(df["dist"].values[:-1] <= df["dist"].values[1:])

def test_plot_lightcurves():
    # Create test data
    primer_dict = {
        "lc_tns_z": 0.1,
        "lc_tns_name": "Test SN",
        "lc_tns_cls": "SN Ia",
        "lc_ztf_id": "ZTF21abbzjeq"
    }
    
    # Test basic plotting
    plot_lightcurves(
        primer_dict=primer_dict,
        plot_label="Test",
        theorized_lightcurve_df=None,
        neighbor_ztfids=["ZTF19aaaaaaa"],
        ann_locus_l=[],  # Will be populated by ANTARES client
        ann_dists=[0.5],
        tns_ann_names=["Test Neighbor"],
        tns_ann_classes=["SN II"],
        tns_ann_zs=[0.2],
        figure_path="../figures",
        save_figures=True
    )
    
    # Verify figure was created
    assert plt.get_fignums()  # Check if any figures exist
    plt.close('all')

def test_plot_hosts():
    # Test host galaxy plotting
    plot_hosts(
        ztfid_ref="ZTF21abbzjeq",
        plot_label="Test Host",
        df=pd.DataFrame({'ztf_id': ['ZTF19aaaaaaa']}),
        figure_path="../figures",
        ann_num=1,
        save_pdf=True,
        imsizepix=100,
        change_contrast=False,
        prefer_color=True
    )
    
    # Verify figure was created
    assert plt.get_fignums()
    plt.close('all')

def test_corner_plot():
    # Create test data
    neighbors_df = pd.DataFrame({
        'mag': np.random.normal(20, 1, 100),
        'color': np.random.normal(0, 0.2, 100),
        'period': np.random.uniform(1, 10, 100)
    })
    
    primer_dict = {
        "lc_tns_z": 0.1,
        "lc_tns_name": "Test SN",
        "lc_tns_cls": "SN Ia",
        "lc_ztf_id": "ZTF21abbzjeq"
    }
    
    # Test corner plot creation
    corner_plot(
        neighbors_df=neighbors_df,
        primer_dict=primer_dict,
        path_to_dataset_bank="../data/dataset_bank.csv",
        remove_outliers_bool=True,
        path_to_figure_directory="../figures",
        save_plots=True
    )
    
    # Verify figure was created
    assert plt.get_fignums()
    plt.close('all')

def test_plot_invalid_input():
    with pytest.raises(ValueError):
        # Test with invalid ZTF ID
        plot_lightcurves(
            primer_dict={"lc_ztf_id": "invalid_id"},
            plot_label="Test",
            theorized_lightcurve_df=None,
            neighbor_ztfids=[],
            ann_locus_l=[],
            ann_dists=[],
            tns_ann_names=[],
            tns_ann_classes=[],
            tns_ann_zs=[],
            figure_path="../figures"
        )
