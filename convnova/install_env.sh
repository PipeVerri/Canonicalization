conda create -n canonicalization -c conda-forge python=3.10 cpython
conda activate canonicalization
conda install -n canonicalization pip
conda install minimap2 ncbi-datasets-cli optuna -c conda-forge -c bioconda
pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install "setuptools>=59.6.0,<68.0.0" # For pkg_resources
pip install -e .
echo "Download the flash attn prebuilt wheel from https://mjunya.com/flash-attention-prebuild-wheels/%"
echo -n "Your python version is 3.10 and torch's cuda version is:"
python -c "import torch; print(torch.version.cuda)"
echo "\nYou should also (probably) login into wandb"
