# syntax=docker/dockerfile:1

# FROM ufoym/deepo:tensorflow-py36-cpu
FROM python:3.8

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./train.py .