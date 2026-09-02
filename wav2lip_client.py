import os
import time

def call_wav2lip_api(audio: bytes) -> bytes:
  import requests
  from io import BytesIO
  host = os.getenv("WAV2LIP_HOST")
  if host is None:
    raise RuntimeError("환경변수 'WAV2LIP_HOST'가 설정되지 않았습니다.")
  url = f"http://{host}/inference"
  # BytesIO 객체로 감싸서 requests에서 업로드할 수 있도록 처리
  audio_file = BytesIO(audio)
  audio_file.name = "input.wav" # 업로드될 파일 이름 설정
  files = {'audio': (audio_file.name, audio_file, 'audio/wav')}
  start_time = time.time()
  response = requests.post(url, files=files)
  end_time = time.time()
  print(f"API 호출 시간: {end_time - start_time:.2f}초")
  if response.status_code == 200:
    return response.content # mp4 binary
  else:
    raise Exception(f":x: API 호출 실패: {response.status_code}, {response.text}")
  
if __name__ == "__main__":
  audio_path = "input_1.wav"
  with open(audio_path, "rb") as audio_file:
    audio_bytes = audio_file.read()
  result = call_wav2lip_api(audio_bytes)
  with open("outputs/result.mp4", "wb") as f:
    f.write(result)
