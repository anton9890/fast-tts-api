import logging, io, wave, os, traceback
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 현재 디렉토리 설정
current_dir = "/workspace/hkkim/fast-tts-api"

def audio_array_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """
    numpy float32 배열을 16-bit mono WAV 포맷의 bytes로 변환합니다.

    Args:
        audio_array (np.ndarray): -1.0 ~ 1.0 범위의 float32 오디오 배열
        sample_rate (int): 샘플레이트 (기본값: 24000Hz)

    Returns:
        bytes: WAV 포맷 오디오 데이터
    """
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16bit PCM = 2 bytes
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_array * 32767).astype(np.int16).tobytes())
        return wav_buffer.getvalue()