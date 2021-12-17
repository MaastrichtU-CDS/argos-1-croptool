FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip3 install --upgrade pip

RUN useradd -m leroy

RUN chown -R leroy:leroy /home/leroy

COPY --chown=leroy . /home/leroy/app/

USER leroy

RUN cd /home/leroy/app/ && pip3 install -r requirements.txt

WORKDIR /home/leroy/app
