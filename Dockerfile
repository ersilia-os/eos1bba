FROM bentoml/model-server:0.11.0-py37
MAINTAINER ersilia

RUN pip install rdkit-pypi==2022.9.5
RUN pip install pandas==1.3.5
RUN pip install networkx==2.6.3
RUN pip install paddlepaddle==2.4.1
RUN pip install pgl==2.2.5
RUN pip install scikit-learn==1.0.2

WORKDIR /repo
COPY . /repo
