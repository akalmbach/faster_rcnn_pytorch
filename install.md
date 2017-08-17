# Install pytorch
```
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" # [anaconda root directory]
conda create -n py2SEAL ipython jupyter numpy pyyaml mkl setuptools cmake gcc cffi matplotlib python=2.7
source activate py2SEAL
conda install -c soumith magma-cuda80 # or magma-cuda75 if CUDA 7.5

git clone https://github.com/pytorch/pytorch.git
cd pytorch
python setup.py install
```

# Install akalmbach/pytorch-faster-rcnn
```
cd ..
conda install pip sympy h5py cython
conda install -c menpo opencv3
pip install easydict

git clone https://github.com/akalmbach/faster_rcnn_pytorch.git
cd faster_rcnn_pytorch/faster_rcnn
./make.sh
```
# You may have to edit the nvcc flag -arch=sm_50 for compatibility with your GPU. See http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

# Download the model into faster_rcnn_pytorch/demo https://drive.google.com/open?id=0B4pXCfnYmG1WOXdpYVFybWxiZFE (or modify demo.py to look for the model elsewhere)

# Test the installation with
`python demo.py`

# To import FasterRCNN in your python script, you must copy pytorch-faster-rcnn/faster_rcnn and make it a subdirectory where your script lives. No library for now because of the way fast_rcnn is packaged inside :(
