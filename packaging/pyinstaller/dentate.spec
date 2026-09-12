# -*- mode: python ; coding: utf-8 -*-
# PyInstaller one-folder bundle of `dentate[demo]` (installed from PyPI): `pyinstaller packaging/pyinstaller/dentate.spec`
# → dist/dentate/dentate(.exe). Built per OS by .github/workflows/bundles.yml and zipped as dentate-<target>.zip.
#
# What must travel with the executable, and why:
#   dentate/demo/**   the offline demo (starter project + pinned SmolLM tokenizer) — `dentate demo init|run`
#   dentate/web/**    the built SPA — `dentate serve`
#   torch/transformers/tokenizers/spiral   heavy, lazily imported, with native libs + data PyInstaller cannot trace
#   dist-info metadata  transformers gates features on importlib.metadata.version(...); dentate discovers Spiral
#                       through the `dentate.models` entry point, which only exists if spiral-lm's metadata ships
from importlib.metadata import PackageNotFoundError

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

datas, binaries, hiddenimports = [], [], []

for package in ("torch", "transformers", "tokenizers", "safetensors", "spiral", "regex", "huggingface_hub"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

datas += collect_data_files("dentate", includes=["demo/**/*", "web/**/*"])

for distribution in ("dentate", "spiral-lm", "torch", "transformers", "tokenizers", "safetensors", "huggingface-hub",
                     "numpy", "packaging", "filelock", "requests", "tqdm", "regex", "pyyaml", "fastapi", "uvicorn",
                     "pydantic", "PyJWT", "cryptography", "httpx", "python-multipart"):
    try:
        datas += copy_metadata(distribution)
    except PackageNotFoundError:
        pass   # optional in some environments (e.g. no cryptography backend); the CLI reports what is missing

# dentate imports most subcommand machinery lazily inside the handlers; walk the package so nothing is left behind.
hiddenimports += collect_submodules("dentate")
hiddenimports += collect_submodules("uvicorn") + collect_submodules("fastapi") + collect_submodules("starlette")
hiddenimports += collect_submodules("pydantic") + ["multipart", "python_multipart", "jwt", "jwt.algorithms", "httpx",
                                                   "yaml", "numpy", "sqlite3", "cryptography.hazmat.backends.openssl"]

a = Analysis(
    ["entry.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    runtime_hooks=[],
    # Apple-only / research-only extras the demo never needs; excluding them keeps the bundle honest and smaller.
    excludes=["mlx", "mlx_lm", "mlx_lm_lora", "peft", "trl", "verl", "datasets", "wandb", "playwright", "pytest",
              "tkinter", "matplotlib", "IPython", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dentate",
    debug=False,
    strip=False,
    upx=False,          # torch's shared libraries do not survive UPX; keep everything as shipped
    console=True,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="dentate")
