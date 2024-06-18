# Use a imagem PyTorch com CUDA e cuDNN como base
FROM  pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Copie os arquivos necessários para o diretório de trabalho do contêiner
COPY . /app

# Defina o diretório de trabalho para o local onde você deseja executar seu código
WORKDIR /app

# Instale as bibliotecas de sistema necessárias, limpe o cache e atualize pip
RUN apt-get update && apt-get install -y libsndfile1  \
    && apt-get clean \
    && apt-get install git -y\
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade git+https://github.com/huggingface/transformers.git

RUN apt-get update && apt-get install git-lfs \
    && git-lfs install