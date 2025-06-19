from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict
from main import handle_story_continue, handle_chapter_summary_with_music

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="ChatStorys AI Server")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get external server URL from environment variable
EXTERNAL_SERVER_URL = os.getenv("EXTERNAL_SERVER_URL", "http://localhost:8000")

# Pydantic models for request/response validation
class StoryRequest(BaseModel):
    user_id: str
    user_message: str
    book_id: str

class ChapterRequest(BaseModel):
    user_id: str
    book_id: str

class MusicRecommendation(BaseModel):
    title: str
    artist: str

class ChapterResponse(BaseModel):
    status: str
    code: int
    summary: Optional[str] = None
    recommended_music: Optional[List[MusicRecommendation]] = None
    message: Optional[str] = None

@app.post("/story/continue")
async def continue_story(request: StoryRequest):
    """소설 계속 쓰기 엔드포인트"""
    try:
        # AI 모델을 사용하여 소설 생성
        result = handle_story_continue(
            user_id=request.user_id,
            user_message=request.user_message,
            book_id=request.book_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/story/chapter/summary_with_music", response_model=ChapterResponse)
async def generate_chapter_summary(request: ChapterRequest):
    """챕터 요약 및 음악 추천 엔드포인트"""
    try:
        # 챕터 요약 및 음악 추천 생성
        result = handle_chapter_summary_with_music(
            user_id=request.user_id,
            book_id=request.book_id
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True  # Enable auto-reload during development
    ) 