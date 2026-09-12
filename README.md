# Dentate RL bootcamp

Public companion to the [`dentate`](https://pypi.org/project/dentate/) package: the Colab notebook and the student guide.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xpitfire/dentate-bootcamp/blob/main/dentate_bootcamp.ipynb)

- `dentate_bootcamp.ipynb`: install `dentate[demo]`, train the starter project on CPU, open the dashboard through Colab's port proxy.
- `BOOTCAMP.md`: the student guide (local, Colab and hosted paths, result files, project schema, troubleshooting).

Local equivalent:

```
pip install "dentate[demo]"
dentate demo init && dentate demo run
dentate serve
```

Hosted site: https://dentate.cortex.a2olabs.com

These files are published from the Dentate repository (`examples/colab/` and `docs/`); edit them there.
