# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /mammography

COPY mammography/requirements/train.in .
COPY mammography/requirements/dicom.in .

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r train.in -r dicom.in

COPY mammography/config ./mammography/config
COPY mammography/src ./mammography/src
