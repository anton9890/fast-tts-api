from pydantic import BaseModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
import os
from utils.utils import logger, current_dir
from utils.utils import traceback
import re
import numpy as np
import torch

class TextRequest(BaseModel):
    text: str
    language: str = "en"
    super_resolution: bool = False


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


def process_tts(text: str, language: str, model: Xtts, gpt_cond_latent, speaker_embedding) -> np.ndarray:
    try:
        sentences = split_into_sentences(text)
        audio_segments = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            audio = model.inference(
                sentence,
                language,
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7,
            )["wav"]

            if not isinstance(audio, np.ndarray):
                logger.warning(f"음성이 numpy array가 아님. type: {type(audio)}, 문장: {sentence}")
                continue

            if audio.ndim == 0 or audio.size == 0:
                logger.warning(f"문장 '{sentence}'에 대한 오디오 결과가 유효하지 않음. shape: {audio.shape}")
                continue

            audio_segments.append(audio)

        if not audio_segments:
            raise ValueError("모든 문장의 오디오 생성에 실패했습니다. audio_segments가 비어 있음.")

        return combine_audio_segments(audio_segments)

    except Exception as e:
        logger.error(f"TTS 처리 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise
    
def split_into_sentences(text: str) -> list:
    """텍스트를 문장 단위로 분리"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def combine_audio_segments(segments: list) -> np.ndarray:
    """여러 오디오 세그먼트를 하나로 결합"""
    if not segments:
        return np.array([])
    return np.concatenate(segments)

def initialize_tts():
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
    

    # 각 파일의 전체 경로를 리스트로 생성
    reference_audio_paths = "./input/tts_input.wav"

    # tts_model.get_conditioning_latents 함수에 리스트 전달
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=reference_audio_paths)
    
    # 기본 스피커 임베딩 계산
    # reference_audio_path = os.path.join(model_path, "/workspace/hkkim/fast-tts-api/models/xtts_v2/input/audio.wav")
    # gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[reference_audio_path])
    
    return tts_model, tts_config, gpt_cond_latent, speaker_embedding