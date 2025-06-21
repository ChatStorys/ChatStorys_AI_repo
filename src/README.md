# 🎨 AI Novel Generator & Music Recommender

> **AI 기반 소설 생성 및 감정 맞춤형 음악 추천 시스템**

사용자의 입력을 바탕으로 AI가 소설을 생성하고, 생성된 내용의 감정을 분석하여 적절한 음악을 추천하는 지능형 창작 도구입니다.

## ✨ 주요 기능

### 📚 AI 소설 생성
- **GPT-4 기반 창작**: 사용자의 지시에 따라 창의적인 소설 내용 생성
- **맥락 유지**: 이전 챕터와 현재 진행상황을 고려한 일관성 있는 스토리텔링
- **실시간 대화**: 채팅 히스토리 기반의 상호작용적 소설 작성

### 🎵 감정 기반 음악 추천
- **KoELECTRA 감정 분석**: 한국어에 특화된 감정 분석 모델 사용
- **코사인 유사도 알고리즘**: 감정과 음악 특성 벡터 간의 유사도 계산
- **다중 감정 처리**: 6가지 감정(기쁨, 슬픔, 분노, 불안, 상처, 당황) 지원

### 💾 데이터 관리
- **MongoDB 기반**: 사용자, 책, 챕터, 음악 데이터 체계적 관리
- **자동 요약**: 챕터 완료 시 자동 요약 생성 및 저장
- **권한 관리**: 사용자별 책 접근 권한 제어

## 🛠 기술 스택

### Backend
- **Python 3.8+**
- **MongoDB** - 데이터베이스
- **PyMongo** - MongoDB 드라이버

### AI & ML
- **OpenAI GPT-4** - 소설 생성
- **KoELECTRA** - 한국어 감정 분석
- **Transformers** - 모델 로딩 및 추론
- **NumPy** - 수치 연산
- **Torch** - 딥러닝 프레임워크

### Others
- **python-dotenv** - 환경변수 관리

## 📁 프로젝트 구조

```
src/
├── main.py                     # 메인 소설 처리 로직
├── api/
│   └── gpt_client.py          # GPT API 클라이언트
├── database/
│   └── db_manager.py          # MongoDB 데이터베이스 관리
├── utils/
│   ├── emotion_analyzer.py    # KoELECTRA 감정 분석
│   └── music_recommender.py   # 음악 추천 시스템
└── README.md                  # 프로젝트 문서
```

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd deep_prj
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경변수 설정
`.env` 파일을 생성하고 다음 내용을 입력하세요:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# MongoDB
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=novel_db

# KoELECTRA Model
KOELECTRA_MODEL_PATH=outputs/koelectra_emotion
HF_EMOTION_API_URL=https://hglww4g5jugd2khs.us-east-1.aws.endpoints.huggingface.cloud
```

### 5. MongoDB 설정
MongoDB가 설치되어 있고 실행 중인지 확인하세요.

### 6. KoELECTRA 모델 준비
감정 분석용 KoELECTRA 모델을 `outputs/koelectra_emotion` 경로에 배치하세요.

## 📖 API 문서

### 소설 계속 쓰기
사용자의 지시에 따라 AI가 소설 내용을 생성합니다.

**함수**: `handle_story_continue(user_id, user_message, book_id)`

**매개변수**:
- `user_id` (str): 사용자 ID
- `user_message` (str): 사용자 메시지 (스토리 방향 지시)
- `book_id` (str): 책 ID

**응답**:
```json
{
    "status": "success",
    "code": 200,
    "message": "소설 저장 완료",
    "prompt": "생성된 소설 내용"
}
```

### 챕터 요약 및 음악 추천
현재 챕터를 요약하고 감정에 맞는 음악을 추천합니다.

**함수**: `handle_chapter_summary_with_music(user_id, book_id)`

**매개변수**:
- `user_id` (str): 사용자 ID
- `book_id` (str): 책 ID

**응답**:
```json
{
    "status": "success",
    "code": 200,
    "summary": "챕터 요약 내용",
    "recommended_music": [
        {
            "title": "음악 제목",
            "artist": "아티스트"
        }
    ]
}
```

## 🎯 사용 예시

### 기본 사용법
```python
from src.main import handle_story_continue, handle_chapter_summary_with_music

# 소설 계속 쓰기
result = handle_story_continue(
    user_id="user123",
    user_message="주인공이 신비로운 숲에 들어갑니다.",
    book_id="book456"
)

print(result["prompt"])  # 생성된 소설 내용

# 챕터 완료 및 음악 추천
summary_result = handle_chapter_summary_with_music(
    user_id="user123",
    book_id="book456"
)

print(summary_result["summary"])  # 챕터 요약
print(summary_result["recommended_music"])  # 추천 음악
```

## 🧠 알고리즘 상세

### Algorithm 1: 음악 추천 시스템
1. **감정 분석**: KoELECTRA로 소설 내용의 감정 분석
2. **특성 벡터 생성**: 감정에 맞는 7차원 음악 특성 벡터 생성
3. **유사도 계산**: 음악 DB의 각 곡과 코사인 유사도 계산
4. **순위 결정**: 유사도 기준으로 정렬하여 최적 음악 추천

### 감정별 음악 특성 매핑
```python
weight_table = {
    'anger':        [0.14, 0.86, 0.95, 0.05, 0.25, 0.75, 0.20],
    'sadness':      [0.82, 0.18, 0.14, 0.86, 0.05, 0.95, 0.05],
    'anxiety':      [0.22, 0.78, 0.92, 0.08, 0.17, 0.83, 0.10],
    'hurt':         [0.75, 0.25, 0.20, 0.80, 0.13, 0.88, 0.05],
    'embarrassment':[0.33, 0.67, 0.89, 0.11, 0.33, 0.67, 0.15],
    'joy':          [0.50, 0.50, 0.09, 0.91, 0.95, 0.05, 0.90]
}
# [acoustic, electronic, aggressive, relaxed, happy, sad, party]
```

## 🗄 데이터베이스 스키마

### Users Collection
```json
{
    "userId": "string",
    "name": "string",
    "password": "string",
    "bookID": ["array of book IDs"],
    "created_at": "datetime"
}
```

### Books Collection
```json
{
    "bookId": "string",
    "title": "string",
    "genre": "string",
    "userId": "string",
    "created_at": "datetime"
}
```

### Chapters Collection
```json
{
    "bookId": "string",
    "chapter_Num": "string",
    "userId": "string",
    "sumChapter": "string",
    "workingFlag": "boolean",
    "musicTitle": "string",
    "composer": "string",
    "created_at": "datetime"
}
```

### Music Collection
```json
{
    "name": "string",
    "artist": "string",
    "mbid": "string",
    "mood_acoustic": "float",
    "mood_electronic": "float",
    "mood_aggressive": "float",
    "mood_relaxed": "float",
    "mood_happy": "float",
    "mood_sad": "float",
    "mood_party": "float"
}


**Made with ❤️ by AI Novel Generator Team**
