#!/usr/bin/env bash
set -e
sudo apt remove -y python3-pip
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
pip install pip --upgrade
pip install pyopenssl --upgrade
pip install "jax[tpu]==0.4.9" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install optax ray dm-haiku tensorboardX fabric decorator
mkdir -p data
wget https://cs.fit.edu/~mmahoney/compression/enwik8.zip -O data/enwik8.zip
wget https://cs.fit.edu/~mmahoney/compression/enwik9.zip -O data/enwik9.zip
unzip data/enwik8.zip -d data/
unzip data/enwik9.zip -d data/