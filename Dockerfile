FROM tensorflow/tensorflow:latest-jupyter

WORKDIR /opt/project
ADD . /opt/project
RUN pip install -r requirements.txt
CMD ["jupyter","notebook","--ip=0.0.0.0","--no-browser","--allow-root"]