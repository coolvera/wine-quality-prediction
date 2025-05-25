FROM python:3.8-slim
RUN pip install pandas scikit-learn sagemaker-training
COPY train.py /opt/ml/code/train.py
ENV SAGEMAKER_PROGRAM train.py
