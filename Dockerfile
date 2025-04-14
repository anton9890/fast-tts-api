FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

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

# 작업 디렉토리 설정
WORKDIR /app

# 프로젝트 복사
COPY . .

# 디렉토리 생성 및 권한
RUN mkdir -p outputs models && chmod -R 777 outputs models

# 포트 노출
EXPOSE 8000

# requirements 설치
RUN pip install -r requirements.txt

# Hugging Face 로그인 + 서버 실행 스크립트 작성
RUN echo '#!/bin/bash\n\
if [ -z "$HF_TOKEN" ]; then\n\
  echo "ERROR: HF_TOKEN 환경변수가 설정되지 않았습니다."\n\
  exit 1\n\
fi\n\
huggingface-cli login --token $HF_TOKEN\n\
python3 main.py' > /app/start.sh && chmod +x /app/start.sh

# 컨테이너 시작 시 실행할 명령
CMD ["/app/start.sh"]
