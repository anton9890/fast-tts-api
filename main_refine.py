import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from utils.utils import logger, current_dir, audio_array_to_wav_bytes, convert_to_webm, traceback
from utils.wav2lip_config import call_wav2lip_api
from utils.xtts_config import TextRequest, process_tts, initialize_tts
from utils.llm_config import get_llm_response, initialize_llm

app = FastAPI()

# 전역 변수로 모델과 설정 저장
tts_model = None
tts_config = None
gpt_cond_latent = None
speaker_embedding = None

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
app.mount("/outputs", StaticFiles(directory=outputs_dir, html=True), name="outputs")

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 초기화"""
    global tts_model, tts_config, gpt_cond_latent, speaker_embedding
    try:
        # outputs 디렉토리 생성
        os.makedirs(outputs_dir, exist_ok=True)
        logger.info(f"outputs 디렉토리 생성/확인: {outputs_dir}")
        
        # TTS 모델 초기화
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
    if not all([tts_model]):
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다.")
    
    start_total = time.time()
    
    try:
        ##############LLM################
        start_llm = time.time()
        llm_response = get_llm_response(request.text)
        llm_time = time.time() - start_llm
        logger.info(f"LLM 처리 시간: {llm_time:.2f}초")
        ##############LLM################

        ##############TTS################
        start_tts = time.time()
        audio_data = process_tts(
            llm_response, 
            request.language, 
            tts_model, 
            gpt_cond_latent, 
            speaker_embedding
        )
        ##############TTS################

        ##############AUDIO_Array->.WAV################
        audio_bytes = audio_array_to_wav_bytes(audio_data)
        
        tts_time = time.time() - start_tts
        logger.info(f"TTS 처리 시간: {tts_time:.2f}초")
        ##############AUDIO_Array->.WAV################

        ##############Talking Face Generation################
        # Wav2Lip API 호출
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_output = os.path.join(outputs_dir, f"temp_output_{timestamp}.mp4")
        output_video = os.path.join(outputs_dir, f"output_{timestamp}.webm")
        start_wav2lip = time.time()
        video_data = call_wav2lip_api(audio_bytes, request.super_resolution)
        wav2lip_time = time.time() - start_wav2lip
        with open(temp_output, 'wb') as f:
            f.write(video_data)
        ##############Talking Face Generation################

        ##############VIDEO_WEBM################
        if not convert_to_webm(temp_output, output_video):
            raise HTTPException(status_code=500, detail="비디오 변환에 실패했습니다.")

        # 임시 파일 삭제
        try:
            os.remove(temp_output)
        except Exception as e:
            logger.warning(f"임시 파일 삭제 실패: {e}")
                    
        logger.info(f"Wav2Lip API 처리 시간: {wav2lip_time:.2f}초")
        
        if not os.path.exists(output_video):
            raise HTTPException(status_code=500, detail="비디오 생성에 실패했습니다.")
        ##############VIDEO_WEBM################
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
            "status": "success",
            "video_url": video_url,
            "llm_response": llm_response,
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
    