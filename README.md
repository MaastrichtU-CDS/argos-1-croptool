# Argos crop

This code requires that your data is already pulled from XNAT and stored in a folder on your host machine. If you have not done so yet, please run the XNAT extraction code first or contact Leonard Wee for instructions. This folder should contain 2 subfolders; pre-process-TRAIN, and pre-process-VALIDATE.

Step 0:
*** If you previously done our ARGOS GPU standalone test you can skip steps 1 and 2, because it was done during that test. ***

Step 1:
If you have not already done so, install an nvidia driver:\
	Step 1: Identify the type of GPU in your system by using the console and typing: ubuntu-drivers devices\
	Step 2: Your GPU and a list of drivers should appear. Please make a printscreen of this output and send it to us. We recommend 				installing the latest recommended version or a version >= 418.81.07.
			For example on a Tesla M60 we would type: sudo apt install nvidia-driver-490\
	Step 3: Reboot your Ubuntu machine.

Step 2:
The next step is to install Docker-CE by typing the commands below or follow the official instructions here (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker):\
curl https://get.docker.com | sh \
&& sudo systemctl --now enable docker

Next we install the NVIDIA Container toolkit:\
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

Update: sudo apt-get update
Install the nvidia docker container: sudo apt-get install -y nvidia-docker2
Restart the docker daemon: sudo systemctl restart docker
We can test if the installation was succesful by typing: sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi


Step 3:
Download and unzip the argos-crop-main.zip\
Change your working directory to the argos-crop-main folder, for example, on my machine it is /home/ubuntu/argos-crop-main :\
```
cd {your_path_here}/argos-crop-main
```

Next we build the container:\
```
sudo docker build -f Dockerfile -t crops .
```

Please read the following instructions fully before executing.\
To run the preprocessing you need to adjust the path in this command to the directory where the folders called "pre-process-TRAIN" and "pre-process-VALIDATE" are found. You would have done these by running Leonard's batch conversion to NRRD script and then extracted them from XNAT (see Leonard's guidance videos):\
```
sudo docker run --gpus all -v {/path/to/your/data}:/home/leroy/app/data -ti crops python3 -u crop_lung_volume.py
```
NB : Change the above "{/path/to/your/data}" to your own actual full path. For example, on my machine it is ```/home/ubuntu/xnat-docker-compose-master/pyradiomics-master/o-raw```\

This script will search for lung slices and only select those for deep learning. It will create them in a new folder called "Train" and "Validate". It will put these at the same location where "pre-process-TRAIN" and "pre-process-VALIDATE" were located. Additionally, it will write 2 csv files (train_list.csv and validation_list.csv) with patient folder names and image shapes (e.g. patient-001, (512, 512, 89)). Please check if these csv files are okay to send, and send them to Leonard Wee.

---
Epoch training and Prediction
---
To run our run_online_epoch.py and predict_full.py scripts, please first COPY your Train and Validation folders from the Vantage6 mount to a directory where you have permissions (e.g. /home/...). These 2 scripts will save 4 .csv files. Please note that both of these scripts can have a runtime of several hours.

Change directory to the folder containing this code and then build the container:
```
sudo docker build -f Dockerfile -t crops .
```

Please run this check first (~5 minutes). This loads all the images and saves 2 csv files with image shapes. Any strange results in this list might need to be removed first.
```
sudo docker run -v {/home/ubuntu/ARGOS_Data}:/home/leroy/app/data -ti crops python3 -u test_loading.py
```

```
sudo docker run --gpus all -v {/home/ubuntu/ARGOS_Data}:/home/leroy/app/data -ti crops python3 -u run_online_epoch.py
```
NB: Please change "{/home/ubuntu/ARGOS_Data}" to your new path.

```
sudo docker run --gpus all -v /home/ubuntu/ARGOS_Data:/home/leroy/app/data -ti crops python3 -u predict_full.py
```




