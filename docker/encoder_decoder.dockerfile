FROM hatespace_base as base

ADD /scripts/encoder_decoder/train.py /scripts/encoder_decoder/train.py

ENTRYPOINT ["python", "/scripts/encoder_decoder/train.py"]