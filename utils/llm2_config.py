from utils.utils import logger
from utils.utils import traceback
from mlx_lm import load, generate

def initialize_llm():
    """Gemma 3 모델 초기화"""
    global model
    global tokenizer
    try:
        logger.info("Gemma 3 모델 초기화 중...")
        model, tokenizer = load("mlx-community/gemma-3-1b-it-4bit")
        logger.info("Gemma 3 모델 초기화 완료")
    except Exception as e:
        logger.error(f"Gemma 3 모델 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def get_llm_response(text: str) -> str:
    """Gemma 3 모델을 사용하여 응답 생성"""
    try:
        if model is None:
            logger.warning("Gemma 3 모델이 초기화되지 않았습니다.")
            return f"I received your message: {text}"

        # 프롬프트 형식 지정
        system_prompt = """You are a friendly and helpful AI assistant who speaks in a natural, conversational way. 
When responding:
- Use a warm and engaging tone
- Keep explanations simple and relatable
- Use everyday language instead of technical jargon when possible
- Include brief examples or analogies when helpful
- Show empathy and enthusiasm in your responses
- Feel free to use casual expressions and contractions (like "I'm", "let's", etc.)
- Keep responses concise but informative

Remember to always be helpful while maintaining a natural, human-like conversation style."""

        prompt = f"""### System: {system_prompt}

### Human: {text}

### Assistant:"""
        
        # 만약 tokenizer에 chat_template이 정의되어 있다면 템플릿 적용
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": text}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
        # Gemma 3 모델로 응답 생성 (generate 함수 사용)
        response = generate(model, tokenizer, prompt=prompt, verbose=True)
        
        # 필요에 따라 추가 전처리 작업 수행 가능
        if not response:
            logger.warning("모델이 빈 응답을 반환했습니다.")
            return f"I received your message: {text}"
            
        return response
            
    except Exception as e:
        logger.error(f"Gemma 3 응답 생성 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return f"I received your message: {text}"

# 예시 사용법:
if __name__ == "__main__":
    initialize_llm()
    user_input = "Hello, how are you?"
    reply = get_llm_response(user_input)
    print("AI 답변:", reply)