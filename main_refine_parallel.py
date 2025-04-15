import torch
import numpy as np
import asyncio
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from utils_parallel.utils import logger
import os
import time
from utils_parallel.utils import traceback, current_dir, convert_to_webm, static_dir, outputs_dir
from utils_parallel.wav2lip_config import wav2lip_consumer
from utils_parallel.xtts_config import TextRequest, process_tts, initialize_tts
from utils_parallel.llm_config import get_llm_response, initialize_llm
from queue import Queue

app = FastAPI()

global tts_model, tts_config, gpt_cond_latent, speaker_embedding, audio_queue

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
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
    global tts_model, tts_config, gpt_cond_latent, speaker_embedding, audio_queue
    try:
        #전역 queue생성
        audio_queue = Queue()

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
    try:
        ############## LLM ################
        start_time = time.time()
        llm_response = get_llm_response(request.text)
        llm_time = time.time()
        logger.info(f"LLM 응답 생성 시간: {llm_time - start_time:.2f}초")

        ############## TTS ################
        # 전역 큐 새로 생성 (request별)
        from queue import Queue
        from threading import Thread
        local_queue = Queue()
        
        # TTS 백그라운드 실행 (blocking하므로 Thread 사용)
        tts_thread = Thread(target=process_tts, args=(
            llm_response,
            request.language,
            tts_model,
            gpt_cond_latent,
            speaker_embedding,
            local_queue
        ))
        tts_thread.start()

        ############## Wav2Lip ################
        timestamp = int(time.time())
        video_filename = f"result_{timestamp}.mp4"
        final_path = os.path.join(outputs_dir, video_filename)

        # wav2lip_consumer는 blocking 함수 → asyncio에서 실행하려면 run_in_executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, wav2lip_consumer, local_queue, request.super_resolution, final_path)

        wav2lip_time = time.time()
        logger.info(f"Wav2Lip 처리 시간: {wav2lip_time - llm_time:.2f}초")

        # 결과 파일 존재 확인
        if not os.path.exists(final_path):
            raise HTTPException(500, "최종 결과 영상이 생성되지 않았습니다.")

        return {
            "status": "success",
            "video_url": f"/outputs/{video_filename}",
            "llm_response": llm_response,
            "processing_times": {
                "llm": round(llm_time - start_time, 2),
                "wav2lip": round(wav2lip_time - llm_time, 2),
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
    
    
# 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