# For more information, please refer to https://aka.ms/vscode-docker-python
FROM public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-pytorch-cuda-full:v1.5.0

USER root

RUN apt-get update && apt-get install -y python3-opencv wget g++

WORKDIR /wd

RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

COPY requirements.txt /wd
RUN pip install -r requirements.txt

COPY pest_detection.py /wd
COPY pest_detection_utils.py /wd
COPY pest_detection_kserve.py /wd
COPY kserve_backbone_pest.py /wd
COPY kserve_utils.py /wd
COPY Detic/ /wd/Detic/
COPY detectron2/ /wd/detectron2/
WORKDIR /wd/detectron2
RUN pip install -e .

WORKDIR /wd/Detic
RUN pip install -r requirements.txt

WORKDIR /wd

COPY models_clip/ /wd/models_clip/
COPY models/ /wd/models/
COPY datasets /wd/datasets/

RUN chmod -R 777 /wd

USER jovyan
#RUN mkdir -p /wd/outputs/face_detection_cache/FaceDetector/


ENTRYPOINT ["python", "-u", "kserve_backbone_pest.py"]
