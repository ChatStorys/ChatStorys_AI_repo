# Hugging Face 배포 모델을 사용한 감정 분석 시스템

이 시스템은 Hugging Face에 배포된 KoELECTRA 모델 `Jinuuuu/KoELECTRA_fine_tunning_emotion`을 사용하여 한국어 텍스트의 감정을 분석하고 음악을 추천합니다.

## 주요 특징

- **Hugging Face 모델 사용**: 온라인 배포된 모델을 직접 로드하여 사용
- **로컬 모델 폴백**: Hugging Face 모델 로드 실패 시 로컬 모델로 자동 전환
- **긴 텍스트 처리**: 512 토큰 초과 텍스트를 청크 단위로 분석하여 평균 결과 제공
- **6가지 감정 분류**: angry, happy, anxious, embarrassed, sad, heartache

## 설치 요구사항

```bash
pip install torch transformers numpy python-dotenv
```

## 사용법

### 1. 기본 감정 분석

```python
from src.utils.emotion_analyzer_hf import EmotionAnalyzer

# Hugging Face 모델 사용
analyzer = EmotionAnalyzer(
    model_name="Jinuuuu/KoELECTRA_fine_tunning_emotion",
    use_local=False
)

# 감정 분석
text = "오늘 정말 행복한 하루였어요!"
emotions = analyzer.analyze_emotion_with_KoELECTRA(text)
print(emotions)
# 출력: {'angry': 0.05, 'happy': 0.85, 'anxious': 0.03, ...}

# 주요 감정 추출
dominant = analyzer.get_dominant_emotion(text)
print(f"주요 감정: {dominant}")  # 주요 감정: happy
```

### 2. 음악 추천 시스템

```python
from src.utils.music_recommender_hf import MusicRecommender

# Hugging Face 모델 사용하여 초기화
recommender = MusicRecommender(use_hf_model=True)

# 소설 내용 기반 음악 특성 벡터 추출
novel_content = "주인공은 깊은 슬픔에 빠져 있었다..."
music_features = recommender._analyze_long_text_music_features(novel_content)
print(f"음악 특성: {music_features}")
```

### 3. 로컬 모델 사용

```python
# 로컬 모델 사용 (기존 방식)
analyzer = EmotionAnalyzer(use_local=True)
recommender = MusicRecommender(use_hf_model=False)
```

## 테스트 실행

```bash
python test_hf_emotion_model.py
```

## 모델 정보

- **모델 이름**: `Jinuuuu/KoELECTRA_fine_tunning_emotion`
- **베이스 모델**: KoELECTRA
- **지원 감정**: 
  - angry (분노)
  - happy (행복)
  - anxious (불안)
  - embarrassed (당황)
  - sad (슬픔)
  - heartache (상처)

## 오류 처리

1. **모델 로드 실패**: Hugging Face 모델 로드 실패 시 자동으로 로컬 모델로 전환
2. **네트워크 오류**: 인터넷 연결 문제 시 로컬 모델 사용
3. **긴 텍스트**: 512 토큰 초과 시 자동으로 청크 단위 분석

## 성능 최적화

- **GPU 사용**: CUDA 사용 가능 시 자동으로 GPU 활용
- **배치 처리**: 여러 텍스트 동시 분석 지원
- **메모리 효율성**: 큰 텍스트를 청크로 분할하여 메모리 사용량 최적화

## 환경 변수

```bash
# .env 파일에 설정
KOELECTRA_MODEL_PATH=outputs/koelectra_emotion  # 로컬 모델 경로 (폴백용)
HF_EMOTION_API_URL=https://hglww4g5jugd2khs.us-east-1.aws.endpoints.huggingface.cloud  # KoELECTRA Inference API
```

## API 참조

### EmotionAnalyzer 클래스

- `analyze_emotion_with_KoELECTRA(text)`: 텍스트 감정 분석
- `get_dominant_emotion(text)`: 주요 감정 반환
- `analyze_emotions(text)`: 음악 추천용 감정 분석
- `test_model()`: 모델 테스트 실행

### MusicRecommender 클래스

- `recommend_music(userID, novelContents, musicDB, N)`: 음악 추천
- `get_music_features_for_emotion(emotion)`: 감정별 음악 특성 벡터
- `cosine_similarity(vecA, vecB)`: 코사인 유사도 계산 