FROM bentoml/model-server:0.11.0-py37
MAINTAINER ersilia

RUN pip install rdkit
RUN pip install numpy
RUN pip install pandas
RUN pip install networkx
RUN pip install paddlepaddle
RUN pip install pgl
RUN pip install sklearn
RUN pip install -U scikit-learn
RUN pip install rdkit-pypi

WORKDIR /repo
COPY . /repo
