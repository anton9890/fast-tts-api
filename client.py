import requests
import io
import time
import wave
import numpy as np
import threading
from queue import Queue
import sounddevice as sd
from requests.exceptions import RequestException, ConnectionError, Timeout

class AudioPlayer:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.audio_queue = Queue()
        self.is_playing = False
        self.stream = None
        
        # 오디오 장치 목록 가져오기
        self.devices = sd.query_devices()
        self.default_device = sd.default.device[1]  # 출력 장치
        
        print("\n사용 가능한 오디오 장치:")
        for i, device in enumerate(self.devices):
            if device['max_output_channels'] > 0:  # 출력 장치만 표시
                print(f"{i}: {device['name']} (출력 채널: {device['max_output_channels']})")
        
        print(f"\n기본 출력 장치: {self.devices[self.default_device]['name']}")
    
    def audio_callback(self, outdata, frames, time, status):
        if self.audio_queue.empty():
            if self.is_playing:
                self.stop()
            outdata.fill(0)
            return
        
        chunk = self.audio_queue.get()
        if len(chunk) < len(outdata):
            outdata[:len(chunk)] = chunk
            outdata[len(chunk):].fill(0)
        else:
            outdata[:] = chunk[:len(outdata)]
    
    def start(self):
        try:
            self.is_playing = True
            self.stream = sd.OutputStream(
                device=self.default_device,
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                dtype=np.float32
            )
            self.stream.start()
        except Exception as e:
            print(f"오디오 장치 오류: {e}")
            print("다른 오디오 장치를 선택해주세요.")
            self.is_playing = False
            raise
    
    def stop(self):
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def add_chunk(self, chunk):
        self.audio_queue.put(chunk)

def check_server_connection():
    """서버 연결 상태 확인"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except (ConnectionError, Timeout):
        return False

def text_to_speech(text: str, language: str = "en", max_retries: int = 3) -> None:
    """텍스트를 음성으로 변환하여 재생"""
    url = "http://localhost:8000/tts"
    
    # JSON 형식으로 데이터 준비
    data = {
        "text": text,
        "language": language
    }
    
    for attempt in range(max_retries):
        try:
            # 서버 연결 확인
            if not check_server_connection():
                print("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
                if attempt < max_retries - 1:
                    print(f"{attempt + 1}번째 재시도 중...")
                    time.sleep(2)
                    continue
                return
            
            print(f"음성 변환 중: '{text}'")
            # JSON 형식으로 데이터 전송
            response = requests.post(
                url,
                json=data,  # json 파라미터 사용
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            # WAV 헤더 처리
            wav_header = response.raw.read(44)
            
            # 오디오 플레이어 초기화
            player = AudioPlayer()
            player.start()
            
            # 청크 단위로 스트리밍 처리
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    audio_chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
                    player.add_chunk(audio_chunk)
            
            # 재생이 끝날 때까지 대기
            while not player.audio_queue.empty() or player.is_playing:
                time.sleep(0.1)
            
            player.stop()
            print("음성 재생 완료")
            return
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            if attempt < max_retries - 1:
                print(f"{attempt + 1}번째 재시도 중...")
                time.sleep(2)
            else:
                print("최대 재시도 횟수를 초과했습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
            return

def save_audio(text: str, output_file: str, language: str = "en", max_retries: int = 3) -> None:
    """텍스트를 음성으로 변환하여 파일로 저장"""
    url = "http://localhost:8000/tts"
    
    # JSON 형식으로 데이터 준비
    data = {
        "text": text,
        "language": language
    }
    
    for attempt in range(max_retries):
        try:
            # 서버 연결 확인
            if not check_server_connection():
                print("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
                if attempt < max_retries - 1:
                    print(f"{attempt + 1}번째 재시도 중...")
                    time.sleep(2)
                    continue
                return
            
            print(f"음성 파일 생성 중: '{text}'")
            # JSON 형식으로 데이터 전송
            response = requests.post(
                url,
                json=data,  # json 파라미터 사용
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"음성이 {output_file}에 저장되었습니다.")
            return
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            if attempt < max_retries - 1:
                print(f"{attempt + 1}번째 재시도 중...")
                time.sleep(2)
            else:
                print("최대 재시도 횟수를 초과했습니다.")
        except Exception as e:
            print(f"오류 발생: {e}")
            return

def interactive_mode():
    """대화형 모드로 실행"""
    print("TTS 클라이언트 대화형 모드")
    print("명령어:")
    print("  t <텍스트>  - 텍스트를 음성으로 변환하여 재생")
    print("  s <텍스트>  - 텍스트를 음성으로 변환하여 파일로 저장")
    print("  q           - 종료")
    
    # 서버 연결 확인
    if not check_server_connection():
        print("\n경고: 서버에 연결할 수 없습니다.")
        print("서버가 실행 중인지 확인해주세요.")
        print("서버를 실행하려면: python main.py")
    
    while True:
        try:
            command = input("\n명령어 입력 (t/s/q): ").strip()
            
            if command == 'q':
                print("프로그램을 종료합니다.")
                break
                
            elif command.startswith('t '):
                text = command[2:].strip()
                if text:
                    text_to_speech(text)
                else:
                    print("텍스트를 입력해주세요.")
                    
            elif command.startswith('s '):
                text = command[2:].strip()
                if text:
                    output_file = f"output_{int(time.time())}.wav"
                    save_audio(text, output_file)
                else:
                    print("텍스트를 입력해주세요.")
                    
            else:
                print("잘못된 명령어입니다. 다시 입력해주세요.")
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 대화형 모드로 실행
    interactive_mode() 