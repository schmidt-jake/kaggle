# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /mammography

COPY mammography/requirements/train.in .

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r train.in

COPY mammography/config ./mammography/config
COPY mammography/src ./mammography/src

ENTRYPOINT [ "python", "-m", "mammography.src.train" ]
