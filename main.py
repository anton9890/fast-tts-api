import torch
import numpy as np
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import io
import wave
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
import os
import sys
import logging
from pydantic import BaseModel
import traceback
import time
from transformers import pipeline
import tempfile
import requests
from pathlib import Path
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 현재 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# IP:PORT만 환경 변수로 받기
WAV2LIP_HOST = os.getenv("WAV2LIP_HOST")

if WAV2LIP_HOST is None:
    raise RuntimeError("환경변수 'WAV2LIP_HOST'가 설정되지 않았습니다. 예: 220.118.109.65:8000")

# 전체 URL 조립
WAV2LIP_API_URL = f"http://{WAV2LIP_HOST}/inference"

print(f"[INFO] Wav2Lip API endpoint: {WAV2LIP_API_URL}")

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
static_dir = os.path.join(current_dir, "static")
outputs_dir = os.path.join(current_dir, "outputs")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 전역 변수로 모델과 설정 저장
tts_model = None
tts_config = None
gpt_cond_latent = None
speaker_embedding = None
llm_pipeline = None

# 문장 분리를 위한 최대 텍스트 길이
MAX_TEXT_LENGTH = 300

class TextRequest(BaseModel):
    text: str
    language: str = "en"
    super_resolution: bool = False

def call_wav2lip_api(audio_bytes: bytes, face_sr: bool = False) -> bytes:
    """Wav2Lip API 호출"""
    try:
        from io import BytesIO
        # API URL에 super resolution 옵션 추가
        url = f"{WAV2LIP_API_URL}?face_sr={'true' if face_sr else 'false'}"
        
        # BytesIO 객체로 감싸서 requests에서 업로드할 수 있도록 처리
        audio_file = BytesIO(audio_bytes)
        audio_file.name = "input.wav"  # 업로드될 파일 이름 설정
        files = {'audio': (audio_file.name, audio_file, 'audio/wav')}
        
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Wav2Lip API 호출 실패: {response.status_code}, {response.text}")
    except Exception as e:
        logger.error(f"Wav2Lip API 호출 중 오류 발생: {e}")
        raise

def initialize_llm():
    """Llama 3 모델 초기화"""
    global llm_pipeline
    try:
        logger.info("Llama 3 모델 초기화 중...")
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        logger.info("Llama 3 모델 초기화 완료")
    except Exception as e:
        logger.error(f"Llama 3 모델 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def get_llm_response(text: str) -> str:
    """Llama 3 모델을 사용하여 응답 생성"""
    try:
        if llm_pipeline is None:
            logger.warning("Llama 3 모델이 초기화되지 않았습니다.")
            return f"I received your message: {text}"

        # 프롬프트 형식 지정
        system_prompt = """You are a friendly and helpful AI assistant who speaks in a natural, conversational way. 
When responding:
- Keep responses concise and to the point
- Use a warm and engaging tone
- Keep explanations simple and relatable
- Use everyday language instead of technical jargon
- Include brief examples when needed (but keep them short)
- Show empathy in your responses
- Use casual expressions and contractions
- Aim for 2-3 sentences per response unless more detail is specifically requested

Remember to always be helpful while keeping responses brief and natural."""

        prompt = f"""### System: {system_prompt}

### Human: {text}

### Assistant: """
        
        # EOS 토큰 ID 설정
        terminators = [
            llm_pipeline.tokenizer.eos_token_id,
            llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,  # 약간 높여서 더 다양한 응답 생성
            top_p=0.9,
        )
        
        # 응답 추출 및 정리
        response = outputs[0]["generated_text"]
        # 프롬프트 제거
        response = response.replace(prompt, "").strip()
        # 다음 대화 턴 제거
        response = response.split("### Human:")[0].strip()
        
        if not response:
            logger.warning("모델이 빈 응답을 반환했습니다.")
            return f"I received your message: {text}"
            
        return response
            
    except Exception as e:
        logger.error(f"Llama 3 응답 생성 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return f"I received your message: {text}"

def download_model():
    """TTS 모델 다운로드 함수"""
    model_path = "models/xtts_v2"
    if not os.path.exists(model_path):
        logger.info("TTS 모델 다운로드 중...")
        snapshot_download(
            "coqui/XTTS-v2",
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        logger.info("TTS 모델 다운로드 완료")
    return model_path

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 초기화"""
    global tts_model, tts_config, gpt_cond_latent, speaker_embedding
    
    try:
        # outputs 디렉토리 생성
        outputs_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        logger.info(f"outputs 디렉토리 생성/확인: {outputs_dir}")
        
        # TTS 모델 다운로드 및 초기화
        model_path = download_model()
        
        # TTS 설정 파일 로드
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
            
        logger.info("TTS 모델 설정 로드 중...")
        tts_config = XttsConfig()
        tts_config.load_json(config_path)
        
        # TTS 모델 초기화
        logger.info("TTS 모델 초기화 중...")
        tts_model = Xtts.init_from_config(tts_config)
        tts_model.load_checkpoint(tts_config, checkpoint_dir=model_path, use_deepspeed=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model.to(device)
        
        # 기본 스피커 임베딩 계산
        reference_audio_path = "input/tts_input.wav"
        gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[reference_audio_path])
        
        # Llama 모델 초기화
        initialize_llm()
        
        logger.info("모든 모델 초기화 완료")
        
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """웹 인터페이스 제공"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    if tts_model is None or tts_config is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다.")
    return {"status": "ok"}

def split_into_sentences(text: str) -> list:
    """텍스트를 문장 단위로 분리"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def combine_audio_segments(segments: list) -> np.ndarray:
    """여러 오디오 세그먼트를 하나로 결합"""
    silence_duration = int(24000 * 0.3)  # 0.3초 무음
    silence = np.zeros(silence_duration)
    
    combined = []
    for segment in segments:
        combined.extend(segment)
        combined.extend(silence)
    
    return np.array(combined)

def process_tts(text: str, language: str, model: Xtts, gpt_cond_latent, speaker_embedding) -> np.ndarray:
    """TTS 처리 함수"""
    if len(text) > MAX_TEXT_LENGTH:
        logger.info(f"텍스트 길이가 {MAX_TEXT_LENGTH}자를 초과하여 문장 단위로 처리합니다.")
        sentences = split_into_sentences(text)
        audio_segments = []
        
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"문장 {i}/{len(sentences)} 처리 중...")
            outputs = model.inference(
                text=sentence,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
            )
            audio_data = outputs['wav']
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            audio_segments.append(audio_data)
        
        return combine_audio_segments(audio_segments)
    else:
        outputs = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
        audio_data = outputs['wav']
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        return audio_data

@app.post("/tts")
async def text_to_speech(request: TextRequest):
    """텍스트를 음성으로 립싱크 비디오로 변환"""
    if not all([tts_model]):
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다.")
    
    start_total = time.time()
    
    try:
        # Llama로 응답 생성
        start_llm = time.time()
        llm_response = get_llm_response(request.text)
        llm_time = time.time() - start_llm
        logger.info(f"LLM 처리 시간: {llm_time:.2f}초")
        
        # TTS 처리
        start_tts = time.time()
        audio_data = process_tts(
            llm_response, 
            request.language, 
            tts_model, 
            gpt_cond_latent, 
            speaker_embedding
        )
        
        # 오디오를 바이트로 변환
        audio_bytes = io.BytesIO()
        with wave.open(audio_bytes, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)
            wav_file.writeframes(audio_data.tobytes())
        
        tts_time = time.time() - start_tts
        logger.info(f"TTS 처리 시간: {tts_time:.2f}초")
        
        # Wav2Lip API 호출
        start_wav2lip = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_video = os.path.join(outputs_dir, f"output_{timestamp}.mp4")
        
        video_data = call_wav2lip_api(audio_bytes.getvalue(), request.super_resolution)
        with open(output_video, 'wb') as f:
            f.write(video_data)
            
        wav2lip_time = time.time() - start_wav2lip
        logger.info(f"Wav2Lip API 처리 시간: {wav2lip_time:.2f}초")
        
        if not os.path.exists(output_video):
            raise HTTPException(status_code=500, detail="비디오 생성에 실패했습니다.")
        
        # 파일 URL 생성
        video_filename = os.path.basename(output_video)
        video_url = f"/outputs/{video_filename}"
        
        total_time = time.time() - start_total
        logger.info(f"\n처리 시간 요약:")
        logger.info(f"- LLM: {llm_time:.2f}초")
        logger.info(f"- TTS: {tts_time:.2f}초")
        logger.info(f"- Wav2Lip API: {wav2lip_time:.2f}초")
        logger.info(f"- 총 소요 시간: {total_time:.2f}초")
        
        return {
            "video_url": video_url,
            "llm_response": llm_response,
            "status": "success",
            "processing_times": {
                "llm": round(llm_time, 2),
                "tts": round(tts_time, 2),
                "wav2lip": round(wav2lip_time, 2),
                "total": round(total_time, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1) 