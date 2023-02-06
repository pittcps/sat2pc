import os
import subprocess

os.system("pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html")
os.system("pip install setuptools==59.5.0")
#os.system("pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113")
os.system("pip install pptk==0.1.0")
os.system("pip install pickle5")    
os.system("pip install matplotlib")
os.system("pip install sympy==1.10.1")
os.system("pip install pyransac3d==0.6.0")
os.system("pip install scipy==1.5.3")
os.system("pip install laspy==1.7.0")
os.system("pip install shapely==2.0.1")
os.system("pip install pyproj==3.1.0")
os.system("pip install alphashape==1.3.1")
os.system("pip install PyYAML==5.3.1")
os.system("pip install imageio==2.9.0")
os.system("pip install gin-config")
os.system("python -m pip install -U scikit-image")
os.chdir('./PyTorchEMD')
os.system("python setup.py install")
os.chdir('../neuralnet-pytorch-master')
os.system("python setup.py install")















