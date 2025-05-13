import logging
import os
import sys
import warnings
from contextlib import contextmanager


@contextmanager
def suppress_output():
    """Temporarily silence stdout, stderr, warnings and logging < CRITICAL."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        logging.disable(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            logging.disable(logging.NOTSET)
            sys.stdout, sys.stderr = old_stdout, old_stderr
