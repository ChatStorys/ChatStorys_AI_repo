import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Union
import os
from dotenv import load_dotenv
import requests

load_dotenv()

class EmotionAnalyzer:
    def __init__(self, model_name: str = None, use_local: bool = False, use_hf_api: bool = True, hf_api_token: str = None, hf_api_url: str = None):
        """
        감정 분석을 위한 KoELECTRA 모델 초기화 (Hugging Face 버전 및 Hosted Inference API 지원)
        
        Args:
            model_name (str): Hugging Face 모델 이름
            use_local (bool): 로컬 모델 사용 여부
            use_hf_api (bool): Hugging Face Hosted Inference API 사용 여부
            hf_api_token (str): Hugging Face API 토큰 (Private 모델일 때 필요)
            hf_api_url (str): Hugging Face Inference API URL (명시적으로 지정 가능)
        """
        self.use_hf_api = use_hf_api
        self.hf_api_token = os.getenv("HF_API_TOKEN")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_hf_api:
            # API URL 지정 또는 환경 변수 사용
            if hf_api_url:
                self.hf_api_url = hf_api_url
            else:
                self.hf_api_url = os.getenv(
                    "HF_EMOTION_API_URL",
                    "https://hglww4g5jugd2khs.us-east-1.aws.endpoints.huggingface.cloud",
                )
            print(f"Hugging Face Inference API 사용: {self.hf_api_url}")
        else:
            if use_local:
                # 로컬 모델 경로 사용
                self.model_path = os.getenv("KOELECTRA_MODEL_PATH", "outputs/koelectra_emotion")
            else:
                # Hugging Face 모델 사용
                self.model_path = model_name or "Jinuuuu/KoELECTRA_fine_tunning_emotion"
            print(f"모델 로드 중: {self.model_path}")
            print(f"사용 디바이스: {self.device}")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                print("모델 로드 완료!")
            except Exception as e:
                print(f"모델 로드 오류: {str(e)}")
                print("로컬 모델로 다시 시도합니다...")
                # Hugging Face 모델 로드 실패 시 로컬 모델로 폴백
                try:
                    local_path = "outputs/koelectra_emotion"
                    self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                    self.model = AutoModelForSequenceClassification.from_pretrained(local_path)
                    self.model.to(self.device)
                    self.model.eval()
                    print("로컬 모델 로드 완료!")
                except Exception as local_error:
                    raise Exception(f"모델 로드 실패 - HF: {str(e)}, Local: {str(local_error)}")
        
        # 감정 레이블 정의 (실제 KoELECTRA 모델 출력 기준)
        self.emotion_labels = [
            'angry', 'anxious', 'embarrassed', 'happy', 'heartache', 'sad'
        ]
        
        # 한국어-영어 감정 매핑 (실제 모델 레이블 기준)
        self.emotion_mapping = {
            '분노': 'angry',
            '행복': 'happy',
            '불안': 'anxious', 
            '당황': 'embarrassed',
            '슬픔': 'sad',
            '상처': 'heartache',
            '기쁨': 'happy',  # 기쁨은 행복으로 매핑
            '기대': 'happy',  # 기대는 행복으로 매핑
            '우울': 'sad'     # 우울은 슬픔으로 매핑
        }

    def analyze_emotion_with_KoELECTRA(self, text: str) -> Dict[str, float]:
        """
        Algorithm 2: KoELECTRA를 이용한 감정 분석 (Hugging Face 버전 및 Hosted Inference API 지원)
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            Dict[str, float]: 감정별 확률값 (probs)
        """
        if not text or not text.strip():
            # 빈 텍스트 처리
            return {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels}
        
        if self.use_hf_api:
            # Hugging Face Hosted Inference API 사용
            headers = {"Content-Type": "application/json"}
            if self.hf_api_token:
                headers["Authorization"] = f"Bearer {self.hf_api_token}"
            data = {"inputs": text}
            try:
                response = requests.post(self.hf_api_url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                # API 응답 파싱 (예상: [{"label": "angry", "score": 0.1}, ...] 또는 logits)
                emotion_probs = {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels}  # 기본값
                if isinstance(result, list) and all("label" in r and "score" in r for r in result):
                    # 예: [{"label": "angry", "score": 0.1}, ...]
                    emotion_probs = {r["label"]: float(r["score"]) for r in result if r["label"] in self.emotion_labels}
                    # 누락된 감정은 0으로 채움
                    for label in self.emotion_labels:
                        if label not in emotion_probs:
                            emotion_probs[label] = 0.0
                elif isinstance(result, dict) and "error" in result:
                    print(f"HF API 오류: {result['error']}")
                elif isinstance(result, list) and len(result) > 0 and "score" in result[0]:
                    # 일부 모델은 [{"label":..., "score":...}] 형태
                    emotion_probs = {r["label"]: float(r["score"]) for r in result if r["label"] in self.emotion_labels}
                elif isinstance(result, dict) and "logits" in result:
                    # 일부 모델은 {"logits": [...]} 형태
                    logits = np.array(result["logits"])
                    probs = np.exp(logits) / np.sum(np.exp(logits))
                    emotion_probs = {label: float(prob) for label, prob in zip(self.emotion_labels, probs)}
                else:
                    print(f"HF API 응답 예외: {result}")
                return emotion_probs
            except Exception as e:
                print(f"HF API 호출 오류: {str(e)}")
                return {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels}
        else:
            try:
                # 1. tokens ← tokenizer.encode(text)
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # 2. outputs ← KoELECTRA_model.predict(tokens)
                with torch.no_grad():
                    outputs = self.model(**tokens)
                
                # 3. probs ← softmax(outputs.logits)
                probs = torch.softmax(outputs.logits, dim=1)
                
                # 4. return probs
                emotion_probs = {}
                
                # 실제 KoELECTRA 모델이 출력하는 순서대로 매핑
                # 모델 출력 순서: angry, anxious, embarrassed, happy, heartache, sad
                for i, emotion_label in enumerate(self.emotion_labels):
                    if i < len(probs[0]):
                        emotion_probs[emotion_label] = float(probs[0][i])
                
                return emotion_probs
                
            except Exception as e:
                print(f"감정 분석 중 오류 발생: {str(e)}")
                # 오류 시 균등 분포 반환
                return {emotion: 1.0/len(self.emotion_labels) for emotion in self.emotion_labels}

    def analyze_emotions(self, text: str) -> Dict:
        """
        음악 추천 시스템을 위한 감정 분석 (main.py에서 사용)
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            Dict: dominant_emotion과 confidence를 포함한 결과
        """
        try:
            # KoELECTRA로 감정 분석
            emotion_probs = self.analyze_emotion_with_KoELECTRA(text)
            
            # 최고 감정과 신뢰도 계산
            if emotion_probs:
                dominant_emotion = max(emotion_probs.items(), key=lambda x: x[1])
                return {
                    'dominant_emotion': dominant_emotion[0],
                    'confidence': dominant_emotion[1],
                    'all_emotions': emotion_probs
                }
            else:
                return {
                    'dominant_emotion': 'happy',
                    'confidence': 0.5,
                    'all_emotions': {'happy': 0.5}
                }
                
        except Exception as e:
            print(f"감정 분석 오류: {str(e)}")
            return {
                'dominant_emotion': 'happy',
                'confidence': 0.5,
                'all_emotions': {'happy': 0.5}
            }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        텍스트의 감정을 분석하여 각 감정의 확률값을 반환 (기존 호환성 유지)
        """
        return self.analyze_emotion_with_KoELECTRA(text)
    
    def analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        여러 텍스트의 감정을 분석
        
        Args:
            texts (List[str]): 분석할 텍스트 리스트
            
        Returns:
            List[Dict[str, float]]: 각 텍스트별 감정 확률값 리스트
        """
        return [self.analyze_emotion_with_KoELECTRA(text) for text in texts]
    
    def get_dominant_emotion(self, text: str) -> str:
        """
        텍스트의 주요 감정을 반환
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            str: 가장 높은 확률을 가진 감정 레이블 (영어)
        """
        emotion_scores = self.analyze_emotion_with_KoELECTRA(text)
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    
    def get_emotion_distribution(self, text: str, top_k: int = 3) -> List[tuple]:
        """
        텍스트의 상위 k개 감정과 확률값을 반환
        
        Args:
            text (str): 분석할 텍스트
            top_k (int): 반환할 감정의 개수
            
        Returns:
            List[tuple]: (감정, 확률) 튜플의 리스트
        """
        emotion_scores = self.analyze_emotion_with_KoELECTRA(text)
        sorted_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:top_k]

if __name__ == "__main__":
    # 테스트용 예시 문장
    test_texts = [
        "오늘은 정말 행복한 하루야!",
        "나는 너무 불안하고 걱정돼.",
        "이별은 항상 마음이 아파.",
        "시험 망쳐서 화가 난다.",
        "친구들 앞에서 실수해서 너무 창피했어.",
        "요즘 너무 우울해.",
        "기대되는 일이 있어!"
    ]

    # Hugging Face Inference API 사용 (기본값 True)
    analyzer = EmotionAnalyzer(model_name="Jinuuuu/KoELECTRA_fine_tunning_emotion")

    for text in test_texts:
        result = analyzer.analyze_emotions(text)
        print(f"문장: {text}")
        print(f"분석 결과: {result}\n")