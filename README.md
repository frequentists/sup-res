# Supervised Research

### Steps to run

We use `python 3.11.9` in this project.

1. Create conda environment

```console
conda create -n sup-res-dl
conda activate sup-res-dl
conda install python=3.11
```

2. Packages

We required `faiss-gpu` for indexing on the gpu, overriding the default behaviour of ragatouille using `plaid`

```console
pip install ragatouille
pip uninstall faiss-cpu
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```
