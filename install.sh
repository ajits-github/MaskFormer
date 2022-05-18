#!/bin/bash

cd /home/akumar/Git/MaskFormer
echo $(pwd)


# conda install -y pytorch=1.10 torchvision torchaudio cudatoolkit=11.3 -c pytorch

# Working:
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
apt update && DEBIAN_FRONTEND=noninteractive apt install -y ffmpeg libsm6 libxext6 git wget apt-transport-https cmake build-essential nano tzdata graphviz unzip p7zip p7zip-full p7zip-rar
pip install pyyaml==5.1
pip install opencv-contrib-python==4.5.4.60
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

pip install -r requirements.txt


