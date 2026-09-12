"""PyInstaller entry point for the downloadable `dentate` CLI (one-folder bundle of `dentate[demo]`)."""

import multiprocessing
import sys

from dentate.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()   # torch/tokenizers may spawn helpers; a frozen child must not re-run the CLI
    sys.exit(main())
