docker run  --shm-size=1024m --ipc=host -it --gpus all --rm -v $(pwd):/home/taiki-kondo/ -e "HOME=$HOME" yolov5-custom bash
