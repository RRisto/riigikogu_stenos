FROM jupyter/scipy-notebook
USER root

RUN apt-get update
RUN apt-get install -y gcc build-essential python-dev

RUN pip install --upgrade pip
RUN mkdir /opt/install
COPY requirements.txt /opt/install/requirements.txt
RUN pip install -r /opt/install/requirements.txt

USER ${NB_UID}
WORKDIR "${HOME}/work"
