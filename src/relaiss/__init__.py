from importlib import metadata
from .relaiss import ReLAISS, load_reference, find_neighbors

try:
    __version__ = metadata.version("relaiss")
except metadata.PackageNotFoundError:  # package not installed
    __version__ = "0.0.0.dev0"

__all__: list[str] = [
    "ReLAISS",
    "load_reference",
    "find_neighbors",
    "__version__",
]
