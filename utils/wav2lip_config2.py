import requests
from utils.utils import logger, WAV2LIP_API_URL

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
