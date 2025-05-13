<p align="center">
  <img src="https://github.com/evan-reynolds/re-laiss/blob/main/static/reLAISS_logo.png" style="width: 50%;" alt="reLAISS Logo">
</p>

_A flexible library for similarity searches of supernovae and their host galaxies._

reLAISS lets you retrieve nearest‑neighbour supernovae (or spot outliers) by combining ZTF $g/r$ light‑curve morphology with Pan‑STARRS host‑galaxy colours. A pre‑built reference index lets you find similar events to a queried object in seconds, and the modularity of the code allows you to customize it for your own science case.

# Install

Installation of the package is easy: 

`pip install relaiss`


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

If reLAISS helps your research, please cite:

```
Research note bibtex to be added here!
```
