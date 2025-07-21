[![Unit Tests](https://github.com/evan-reynolds/re-laiss/actions/workflows/ci.yml/badge.svg)](https://github.com/evan-reynolds/re-laiss/actions/workflows/ci.yml)

<p align="center">
  <img src="https://github.com/evan-reynolds/re-laiss/blob/main/static/reLAISS_logo.png" style="width: 50%;" alt="reLAISS Logo">
</p>

<p align="center">
  <em>A flexible library for similarity searches of supernovae and their host galaxies.</em>
</p>

reLAISS lets you retrieve nearest‑neighbour supernovae (or spot outliers) by combining ZTF $g/r$ light‑curve morphology with Pan‑STARRS host‑galaxy colours. A pre‑built reference index allows users find similar events to a queried object in seconds. reLAISS is designed to be modular; feel free to customize for your own science!

# Install

In a fresh conda environment, run `pip install relaiss`. After installing, you must install the python bindings for `ngt` from source: 

```
git clone https://github.com/yahoojapan/NGT.git
cd NGT/python
pip3 install .
```

# Code Demo
```
import relaiss as rl

client = rl.ReLAISS()

# load reference data
client.load_reference(
    path_to_sfd_folder='./sfddata-master',  # Directory for SFD dust maps
    weight_lc=3, # Upweight lightcurve features for neighbor search
)

# Find the 5 closest matches to a ZTF transient
neigh = client.find_neighbors(
        ztf_object_id='ZTF21abbzjeq',  # Using the test transient
        n=5,  # number of neighbors to retrieve
        plot=True, # plot and save figures
        save_figures=True,
        path_to_figure_directory='./figures'
    )

# print closest neighbors and their distances
print(neigh[["iau_name", "dist"]])
```

# Citation

If you use reLAISS for your research, please cite the following two works:

```
@article{Reynolds_2025,
doi = {10.3847/2515-5172/adef56},
url = {https://dx.doi.org/10.3847/2515-5172/adef56},
year = {2025},
month = {jul},
publisher = {The American Astronomical Society},
volume = {9},
number = {7},
pages = {189},
author = {Reynolds, E. and Gagliano, A. and Villar, V. A.},
title = {reLAISS: A Python Package for Flexible Similarity Searches of Supernovae and Their Host Galaxies},
journal = {Research Notes of the AAS},
}


@ARTICLE{2024ApJ...974..172A,
       author = {{Aleo}, P.~D. and {Engel}, A.~W. and {Narayan}, G. and {Angus}, C.~R. and {Malanchev}, K. and {Auchettl}, K. and {Baldassare}, V.~F. and {Berres}, A. and {de Boer}, T.~J.~L. and {Boyd}, B.~M. and {Chambers}, K.~C. and {Davis}, K.~W. and {Esquivel}, N. and {Farias}, D. and {Foley}, R.~J. and {Gagliano}, A. and {Gall}, C. and {Gao}, H. and {Gomez}, S. and {Grayling}, M. and {Jones}, D.~O. and {Lin}, C. -C. and {Magnier}, E.~A. and {Mandel}, K.~S. and {Matheson}, T. and {Raimundo}, S.~I. and {Shah}, V.~G. and {Soraisam}, M.~D. and {de Soto}, K.~M. and {Vicencio}, S. and {Villar}, V.~A. and {Wainscoat}, R.~J.},
        title = "{Anomaly Detection and Approximate Similarity Searches of Transients in Real-time Data Streams}",
      journal = {\apj},
     keywords = {Supernovae, Transient detection, Astronomical methods, Time domain astronomy, Time series analysis, Astrostatistics techniques, Classification, Light curves, Random Forests, 1668, 1957, 1043, 2109, 1916, 1886, 1907, 918, 1935, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = oct,
       volume = {974},
       number = {2},
          eid = {172},
        pages = {172},
          doi = {10.3847/1538-4357/ad6869},
archivePrefix = {arXiv},
       eprint = {2404.01235},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...974..172A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
