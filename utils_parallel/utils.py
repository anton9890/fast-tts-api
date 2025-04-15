import logging, io, wave, os, traceback
import numpy as np
import subprocess

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 현재 디렉토리 설정
current_dir = "/workspace/hkkim/fast-tts-api"

static_dir = os.path.join(current_dir, "static")
outputs_dir = os.path.join(current_dir, "outputs")

WAV2LIP_API_URL = "http://220.118.109.65:8000/inference"

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
    
    
def convert_to_webm(input_video: str, output_video: str) -> bool:
    """비디오를 WebM 형식으로 변환"""
    try:
        command = [
            'ffmpeg', '-i', input_video,
            '-c:v', 'libvpx-vp9',  # VP9 비디오 코덱
            '-c:a', 'libopus',     # Opus 오디오 코덱
            '-b:v', '1M',          # 비디오 비트레이트
            '-b:a', '128k',        # 오디오 비트레이트
            '-cpu-used', '4',        # 인코딩 속도/품질 트레이드오프
            '-f', 'webm',            # WebM 컨테이너
            '-y',                    # 기존 파일 덮어쓰기
            output_video
        ]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"비디오 변환 중 오류 발생: {e}")
        return False
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}")
        return False