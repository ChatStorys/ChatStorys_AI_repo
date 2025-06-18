from typing import Dict, List
from openai import OpenAI
import os
from dotenv import load_dotenv
# from ..utils.prompt_templates import PromptTemplates

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from utils.prompt_templates import PromptTemplates

load_dotenv()

class GPTClient:
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize GPT client for novel generation with enhanced chat history support
        
        Args:
            api_key: OpenAI API key
            model: Model name to use (default from env or gpt-4)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize OpenAI client with new API
        self.client = OpenAI(api_key=self.api_key)
        
        # Model configuration from environment or default
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1500"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Chat history management settings
        self.max_history_tokens = int(os.getenv("MAX_HISTORY_TOKENS", "8000"))  # 히스토리 토큰 한도
        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "16000"))  # 전체 컨텍스트 한도

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough estimation: 1 token ≈ 3.5 characters for Korean/English mix)
        """
        return max(1, len(text) // 3)

    def _prepare_chat_history(self, messages: List[Dict], current_user_message: str, system_prompt: str) -> List[Dict]:
        """
        Prepare chat history while staying within token limits
        최대한 많은 히스토리를 유지하면서 토큰 한도를 준수
        
        Args:
            messages: 채팅 히스토리
            current_user_message: 현재 사용자 메시지
            system_prompt: 시스템 프롬프트
            
        Returns:
            토큰 한도 내의 최적화된 메시지 리스트
        """
        if not messages:
            return []
        
        # 토큰 사용량 계산
        system_tokens = self._estimate_tokens(system_prompt)
        current_tokens = self._estimate_tokens(current_user_message)
        available_tokens = self.max_context_length - self.max_tokens - system_tokens - current_tokens - 500  # 여유분
        
        chat_history = []
        total_tokens = 0
        
        # 최신 메시지부터 역순으로 추가 (최신 컨텍스트 우선 보존)
        for msg in reversed(messages):
            msg_tokens = 0
            temp_messages = []
            
            # 사용자 메시지와 AI 응답을 세트로 처리
            if "User" in msg and "LLM_Model" in msg:
                user_content = msg["User"]
                assistant_content = msg["LLM_Model"]
                
                user_tokens = self._estimate_tokens(user_content)
                assistant_tokens = self._estimate_tokens(assistant_content)
                msg_tokens = user_tokens + assistant_tokens
                
                if total_tokens + msg_tokens <= available_tokens:
                    temp_messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content}
                    ]
                else:
                    # 토큰 한도 초과시 중단
                    break
                    
            elif "User" in msg:
                user_content = msg["User"]
                msg_tokens = self._estimate_tokens(user_content)
                
                if total_tokens + msg_tokens <= available_tokens:
                    temp_messages = [{"role": "user", "content": user_content}]
                else:
                    break
                    
            elif "LLM_Model" in msg:
                assistant_content = msg["LLM_Model"]
                msg_tokens = self._estimate_tokens(assistant_content)
                
                if total_tokens + msg_tokens <= available_tokens:
                    temp_messages = [{"role": "assistant", "content": assistant_content}]
                else:
                    break
            
            # 메시지 추가 (앞쪽에 삽입하여 시간순 유지)
            for temp_msg in reversed(temp_messages):
                chat_history.insert(0, temp_msg)
            total_tokens += msg_tokens
        
        return chat_history

    def chat_session(self, chapter_num: str, context: Dict, user_message: str, messages: List[Dict] = None) -> str:
        """
        Generate novel content using comprehensive chat history and optimized prompts
        
        Args:
            chapter_num: 현재 챕터 번호
            context: 소설 컨텍스트 (이전 챕터, 책 정보 등)
            user_message: 사용자 입력
            messages: 채팅 히스토리 (최대한 많이 활용)
            
        Returns:
            생성된 소설 내용
        """
        try:
            # PromptTemplates에서 전문적인 프롬프트 가져오기
            system_prompt = PromptTemplates.get_chapter_prompt(chapter_num, context)
            
            # 메시지 구성 시작
            chat_messages = [{"role": "system", "content": system_prompt}]
            
            # 채팅 히스토리 최대한 많이 추가 (토큰 한도 내에서)
            if messages:
                history_messages = self._prepare_chat_history(messages, user_message, system_prompt)
                chat_messages.extend(history_messages)
                print(f"📚 Chat history: {len(history_messages)} messages loaded")
            
            # 현재 사용자 메시지 추가
            chat_messages.append({"role": "user", "content": user_message})
            
            # 소설 생성에 최적화된 파라미터로 요청
            response = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                presence_penalty=0.3,   # 새로운 아이디어 생성 장려
                frequency_penalty=0.2,  # 반복 감소
                top_p=0.95,            # 창의성을 위한 nucleus sampling
                stream=False
            )
            
            generated_content = response.choices[0].message.content
            print(f"✅ Generated {len(generated_content)} characters")
            return generated_content
            
        except Exception as e:
            raise Exception(f"Error generating novel content: {str(e)}")

    def summarize_chapter(self, content: str, chapter_num: str = None) -> str:
        """
        Generate comprehensive chapter summary using enhanced prompt template
        
        Args:
            content: 챕터 내용
            chapter_num: 챕터 번호 (선택사항)
            
        Returns:
            전문적인 챕터 요약
        """
        try:
            # PromptTemplates에서 전문적인 요약 프롬프트 사용
            system_prompt = PromptTemplates.get_summary_prompt(content)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.3,  # 요약은 일관성이 중요
                max_tokens=400,   # 더 상세한 요약을 위해 증가
                top_p=0.8
            )
            
            summary = response.choices[0].message.content
            print(f"📝 Chapter summary generated: {len(summary)} characters")
            return summary
            
        except Exception as e:
            raise Exception(f"Error summarizing chapter: {str(e)}")