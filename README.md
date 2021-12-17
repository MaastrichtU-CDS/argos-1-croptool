# Argos dockerized

This code requires that your data is already pulled from XNAT and stored in a folder on your host machine. If you have not done so yet, please run the XNAT extraction code first or contact Leonard Wee for instructions. This folder should contain 2 subfolders; pre-process-TRAIN, and pre-process-VALIDATE.

Step 0:
If you previously participated in our ARGOS GPU test you can skip steps 1 and 2.

Step 1:  
If you have not already done so, install an nvidia driver:  
	Step 1: Identify the type of GPU in your system by using the console and typing: ubuntu-drivers devices  
	Step 2: Your GPU and a list of drivers should appear. Please make a printscreen of this output and send it to us. We recommend 				installing the latest recommended version or a version >= 418.81.07.  
			For example on a Tesla M60 we would type: sudo apt install nvidia-driver-490  
	Step 3: Reboot your Ubuntu machine.

Step 2:  
The next step is to install Docker-CE by typing the commands below or follow the official instructions here (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker):   
curl https://get.docker.com | sh \
&& sudo systemctl --now enable docker

Next we install the NVIDIA Container toolkit:  
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

Update: sudo apt-get update  
Install the nvidia docker container: sudo apt-get install -y nvidia-docker2  
Restart the docker daemon: sudo systemctl restart docker  
We can test if the installation was succesful by typing: sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi  


Step 3:  
Download and unzip the argos-crop-main.zip  
Change your working directory to the argos-crop-main folder: cd your_path_here/argos-crop-main  
Next we build the container: sudo docker build -f Dockerfile -t crops .  

Run on GPU:
Please read the following instructions fully before copying.
To run the preprocessing you need to adjust the path in this command to the path where your data is stored:
sudo docker run --gpus all -v /path/to/your/data:/home/leroy/app/data -ti crops python3 -u crop_lung_volume.py

For example if my data is stored in the following path: /home/ubuntu/Leroy_test/test_data , I would type the command as follows:
sudo docker run --gpus all -v /home/ubuntu/Leroy_test/test_data:/home/leroy/app/data -ti crops python3 -u crop_lung_volume.py

