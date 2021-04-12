# sh!

conda install -c anaconda ipykernel

python -m ipykernel install --user --name=tf_2.4.1_cpu

pip3 install tensorflow==2.4.1 opencv-python==4.5.1.48 pillow=8.2.0 matplotlib==3.4.1


# Proto


wget https://github.com/protocolbuffers/protobuf/releases/download/v3.15.7/protobuf-all-3.15.7.tar.gz
tar xzf protobuf-all-3.15.7.tar.gz
cd protobuf-3.15.7
sudo apt-get update
sudo apt-get install build-essential
sudo ./configure
sudo make
sudo make check
sudo make install 
sudo ldconfig
protoc --version

# compile protos from models tensorflow
# git clone https://github.com/tensorflow/models.git

cd models/research
protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .


#torchvision==0.9.1
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch

# opencv error
# apt-get install ffmpeg libsm6 libxext6  -y

pip3 install Cython
pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

