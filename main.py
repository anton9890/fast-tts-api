import torch
import numpy as np
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import io
import wave
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
import os
import logging
from pydantic import BaseModel
import traceback
import time
from transformers import pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 전역 변수로 모델과 설정 저장
tts_model = None
tts_config = None
gpt_cond_latent = None
speaker_embedding = None
llm_pipeline = None

class TextRequest(BaseModel):
    text: str
    language: str = "en"

def initialize_llm():
    """Llama 모델 초기화"""
    global llm_pipeline
    try:
        logger.info("Llama 모델 초기화 중...")
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info("Llama 모델 초기화 완료")
    except Exception as e:
        logger.error(f"Llama 모델 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def get_llm_response(text: str) -> str:
    """Llama 모델을 사용하여 응답 생성"""
    try:
        if llm_pipeline is None:
            logger.warning("Llama 모델이 초기화되지 않았습니다.")
            return f"I received your message: {text}"

        # 프롬프트 형식 수정
        prompt = f"### Human: {text}\n### Assistant:"
        
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=1,
        )
        
        # 응답 추출 및 정리
        if isinstance(outputs, list) and len(outputs) > 0:
            response = outputs[0]['generated_text']
            # 입력 프롬프트 제거
            response = response.replace(prompt, "").strip()
            # 추가 Human/Assistant 마커 제거
            response = response.split("### Human:")[0].strip()
            return response
        else:
            logger.warning("Llama 모델이 빈 응답을 반환했습니다.")
            return f"I received your message: {text}"
            
    except Exception as e:
        logger.error(f"Llama 응답 생성 중 오류 발생: {e}")
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
        # TTS 모델 다운로드
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
        
        # GPU 사용 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"사용 중인 장치: {device}")
        tts_model.to(device)
        
        # 기본 스피커 임베딩 계산
        reference_audio_path = os.path.join(model_path, "input/input_1.wav")
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"기본 스피커 오디오 파일을 찾을 수 없습니다: {reference_audio_path}")
            
        logger.info("스피커 임베딩 계산 중...")
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
    if tts_model is None or tts_config is None or gpt_cond_latent is None or speaker_embedding is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다.")
    return {"status": "ok"}

@app.post("/tts")
async def text_to_speech(request: TextRequest):
    """텍스트를 음성으로 변환"""
    if tts_model is None or tts_config is None or gpt_cond_latent is None or speaker_embedding is None:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다.")
    
    try:
        # Llama로 응답 생성
        logger.info(f"Llama 응답 생성 시작: text='{request.text}'")
        llm_response = get_llm_response(request.text)
        logger.info(f"Llama 응답: {llm_response}")
        
        # TTS로 음성 생성
        logger.info("음성 생성 시작")
        t0 = time.time()
        
        outputs = tts_model.inference(
            text=llm_response,  # Llama 응답을 TTS 입력으로 사용
            language=request.language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding
        )
        
        # WAV 파일 생성
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)  # 샘플링 레이트를 24000으로 고정
            
            # 오디오 데이터를 WAV 파일에 기록
            audio_data = outputs['wav']
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            elif isinstance(audio_data, np.ndarray):
                audio_data = audio_data.squeeze()  # 불필요한 차원 제거
            
            # 오디오 데이터 정규화 및 변환
            audio_data = np.clip(audio_data, -1.0, 1.0)  # 값 범위 제한
            audio_data = (audio_data * 32767).astype(np.int16)
            wav_file.writeframes(audio_data.tobytes())
        
        wav_buffer.seek(0)
        logger.info(f"음성 생성 완료 (소요 시간: {time.time() - t0:.2f}초)")
        
        # 스트리밍 응답 반환
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav",
                "X-LLM-Response": llm_response  # Llama 응답을 헤더에 포함
            }
        )
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=1
    ) 