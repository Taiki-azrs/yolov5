FROM ultralytics/yolov5
ENV DEBIAN_FRONTEND=noninteractive
RUN pip install wandb
RUN apt update
RUN apt install emacs -y