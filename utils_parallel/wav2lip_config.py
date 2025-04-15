import requests
from utils_parallel.utils import logger, WAV2LIP_API_URL, outputs_dir, convert_to_webm
from queue import Queue
from fastapi import HTTPException
import os

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
    
def wav2lip_consumer(audio_queue: Queue, face_sr: bool = False, final_path: str=None):
    parts = []
    while True:
        idx, audio_bytes = audio_queue.get()
        if idx is None:
            # 모든 문장 처리 완료
            break
        # call Wav2Lip
        video_bytes = call_wav2lip_api(audio_bytes, face_sr)
        # video_bytes → 임시파일 e.g. f"temp_video_{idx}.mp4" 에 저장
        temp_path = os.path.join(os.path.dirname(final_path), f"temp_video_{idx}.mp4")
        with open(temp_path, "wb") as f:
            f.write(video_bytes)
        parts.append(temp_path)

    # 여기서 parts 리스트의 mp4 파일들을 ffmpeg concat → final.mp4
    concat_videos_with_ffmpeg(parts, final_path)
    # 결과 비디오를 파일로 저장
    video_path = os.path.join(outputs_dir, final_path)
    webm_path = os.path.join(outputs_dir, final_path.replace(".mp4", ".webm"))
    
    # WebM 형식으로 변환
    if not convert_to_webm(video_path, webm_path):
        raise HTTPException(status_code=500, detail="비디오 변환에 실패했습니다.")

def concat_videos_with_ffmpeg(part_list, output_path="final.mp4"):
    # 1) list.txt 생성
    #    각 줄에는 `file '파일이름'` 형태로 작성
    with open("concat_list.txt", "w", encoding="utf-8") as f:
        for p in part_list:
            f.write(f"file '{p}'\n")

    # 2) ffmpeg concat 명령 실행
    import subprocess
    cmd = [
        "ffmpeg", "-y", 
        "-f", "concat",
        "-safe", "0",
        "-i", "concat_list.txt",
        "-c", "copy",  # 재인코딩 없이 그대로 복사
        output_path
    ]
    subprocess.run(cmd, check=True)