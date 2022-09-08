FROM python:3.10-slim as base

ADD /hatespace /repo/hatespace
ADD /setup.py /repo/setup.py
ADD /requirements.txt /repo/requirements.txt
WORKDIR /repo

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install .