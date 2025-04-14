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
from utils.utils import logger
import os
import time
from utils.utils import traceback, current_dir, audio_array_to_wav_bytes
from utils.wav2lip_config import call_wav2lip_api
from utils.xtts_config import TextRequest, process_tts, initialize_tts
from utils.llm_config import get_llm_response, initialize_llm

app = FastAPI()

global tts_model, tts_config, gpt_cond_latent, speaker_embedding

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

# 문장 분리를 위한 최대 텍스트 길이
MAX_TEXT_LENGTH = 300

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 초기화"""
    global tts_model, tts_config, gpt_cond_latent, speaker_embedding
    try:
        # outputs 디렉토리 생성
        os.makedirs(outputs_dir, exist_ok=True)
        logger.info(f"outputs 디렉토리 생성/확인: {outputs_dir}")
        
        tts_model, tts_config, gpt_cond_latent, speaker_embedding = initialize_tts()
        # LLM 초기화
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


@app.post("/tts")
async def text_to_speech(request: TextRequest):
    """텍스트를 음성으로 변환하고 Wav2Lip으로 처리"""
    try:
        ##############LLM################
        start_time = time.time()
        llm_response = get_llm_response(request.text)
        llm_time = time.time()
        
        logger.info(f"LLM 응답 생성 시간: {llm_time - start_time:.2f}초")
        ##############LLM################
        
        ##############TTS################
        # TTS로 음성 생성
        audio_array = process_tts(
            llm_response,
            request.language,
            tts_model,
            gpt_cond_latent,
            speaker_embedding
        )
        tts_time = time.time()
        logger.info(f"TTS 처리 시간: {tts_time - llm_time:.2f}초")
        ##############TTS################
        
        ##############AUDIO_Array->.WAV################
        audio_bytes = audio_array_to_wav_bytes(audio_array=audio_array)
        
        ##############Talking Face Generation################
        video_bytes = call_wav2lip_api(audio_bytes, face_sr=request.super_resolution)
        wav2lip_time = time.time()
        logger.info(f"Wav2Lip 처리 시간: {wav2lip_time - tts_time:.2f}초")
        ##############Talking Face Generation################
        
        # 결과 비디오를 파일로 저장
        timestamp = int(time.time())
        video_filename = f"result_{timestamp}.mp4"
        video_path = os.path.join(outputs_dir, video_filename)
        
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        
        return {
            "status": "success",
            "video_url": f"/outputs/{video_filename}",
            "llm_response": llm_response,
            "processing_times": {
                "llm": round(llm_time - start_time, 2),
                "tts": round(tts_time - llm_time, 2),
                "wav2lip": round(wav2lip_time - tts_time, 2),
                "total": round(wav2lip_time - start_time, 2),
            }
        }
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    

#docker run -d -e HF_TOKEN={huggingface_token} -e WAV2LIP_HOST={WAV2LIP_HOST} --gpus all -p 8000:8000 princesslucy/fast-tts-api:latest