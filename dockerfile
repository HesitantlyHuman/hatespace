FROM python:3.10-slim as base

ADD /data /repo/data
ADD /hatespace /repo/hatespace
ADD /scripts /repo/scripts
ADD /setup.py /repo/setup.py
ADD /README.md /repo/README.md
ADD /requirements.txt /repo/requirements.txt
WORKDIR /repo

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install $(cat requirements.txt | grep numpy)
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install .

RUN python3 /repo/scripts/docker/load_transformers.py

ENTRYPOINT [ "/bin/bash" ]