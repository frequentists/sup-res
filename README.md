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

We required `faiss-gpu` for indexing on the gpu, overriding the default behaviour of ragatouille using `plaid`.
The package `faiss-cpu` is installed with ragatouille, but due to a bug in the ragatouille package, it overrides
the `faiss-gpu` package when using a GPU. Thus, it must be uninstalled to force ragatouille to use the GPU.

```console
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install ragatouille
pip uninstall faiss-cpu
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

### Compatibility

Training or indexing with ColBERT on the GPU is currently only compatible with Linux (x86_64 only) for CUDA 12.1.

### Warnings and Errors

Note, due to a known bug in the ragatouille package, when finetuning the colbert model a message might be printed to the console,
that the GPU is not being used despite it being used. The message can just be ignored without aslong as you have a valid GPU setup.

You can check if you have a valid GPU instance of torch by running the following in the console:
```console
python -c "import torch; print(torch.cuda.is_available());print(torch.cuda.get_device_name(torch.cuda.current_device()))"
```
