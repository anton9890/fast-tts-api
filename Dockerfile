FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화 + huggingface-cli 설치
RUN pip3 install --upgrade pip && pip install huggingface_hub

# Hugging Face 로그인
RUN huggingface-cli login --token $HF_TOKEN

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 복사
COPY . .

# 디렉토리 생성 및 권한
RUN mkdir -p outputs models && chmod -R 777 outputs models

# 포트 노출
EXPOSE 8001

# requirements 설치
RUN pip install -r requirements.txt

# 서버 실행
CMD ["python3", "main.py"]
