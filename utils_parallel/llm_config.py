from utils.utils import logger
from transformers import pipeline
import torch
from utils.utils import traceback

def initialize_llm():
    """Llama 3 모델 초기화"""
    global llm_pipeline
    try:
        logger.info("Llama 3 모델 초기화 중...")
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        
        llm_pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        logger.info("Llama 3 모델 초기화 완료")
    except Exception as e:
        logger.error(f"Llama 3 모델 초기화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        raise

def get_llm_response(text: str) -> str:
    """Llama 3 모델을 사용하여 응답 생성"""
    try:
        if llm_pipeline is None:
            logger.warning("Llama 3 모델이 초기화되지 않았습니다.")
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

### Assistant: """
        
        # EOS 토큰 ID 설정
        terminators = [
            llm_pipeline.tokenizer.eos_token_id,
            llm_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,  # 약간 높여서 더 다양한 응답 생성
            top_p=0.9,
        )
        
        # 응답 추출 및 정리
        response = outputs[0]["generated_text"]
        # 프롬프트 제거
        response = response.replace(prompt, "").strip()
        # 다음 대화 턴 제거
        response = response.split("### Human:")[0].strip()
        
        if not response:
            logger.warning("모델이 빈 응답을 반환했습니다.")
            return f"I received your message: {text}"
            
        return response
            
    except Exception as e:
        logger.error(f"Llama 3 응답 생성 중 오류 발생: {e}")
        logger.error(traceback.format_exc())
        return f"I received your message: {text}"