<p align="center">
  <img src="https://github.com/evan-reynolds/re-laiss/blob/main/static/reLAISS_logo.png" style="width: 50%;" alt="reLAISS Logo">
</p>

<p align="center">
  <em>A flexible library for similarity searches of supernovae and their host galaxies.</em>
</p>

reLAISS lets you retrieve nearest‑neighbour supernovae (or spot outliers) by combining ZTF $g/r$ light‑curve morphology with Pan‑STARRS host‑galaxy colours. A pre‑built reference index lets you find similar events to a queried object in seconds, and the modularity of the code allows you to customize it for your own science case.

# Install

Installation of the package is easy: In a fresh conda environment, run `pip install relaiss`


# Code Demo
```
import relaiss as rl

# Load the shipped reference index (~20 000 objects)
index = rl.load_reference()

# Find the 5 closest matches to a ZTF transient
neigh = rl.find_neighbors(
    "ZTF23abcxyz",  # ZTFID
    k=5,             # number of neighbours
    use_lightcurve=True,
    use_host=True,
    host_weight=0.3, # balance LC vs host similarity
    return_dataframe=True,
)
print(neigh[["ztfid", "distance"]])
```

# Citation

If reLAISS helps your research, please cite the following two works:

```
Research note bibtex to be added here!

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
