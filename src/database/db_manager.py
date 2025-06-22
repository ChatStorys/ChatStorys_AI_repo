from typing import Optional, List, Dict
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self, connection_string: str = None):
        # 환경변수에서 MongoDB 연결 문자열 가져오기
        self.connection_string = connection_string or os.getenv("MONGODB_URL")
        if not self.connection_string:
            raise ValueError("MongoDB connection string is required. Please set MONGODB_URL environment variable.")
        
        # MongoDB 연결 설정 개선
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,  # 5초 타임아웃
                connectTimeoutMS=10000,  # 10초 연결 타임아웃
                retryWrites=True
            )
            # 연결 테스트
            self.client.admin.command('ping')
        except Exception as e:
            raise Exception(f"Failed to connect to MongoDB: {str(e)}")
        
        # 데이터베이스 이름을 환경변수에서 가져오거나 기본값 사용
        db_name = os.getenv("MONGODB_DATABASE", "novel_db")
        self.db = self.client[db_name]
        
        # 컬렉션 초기화
        self.users = self.db.users
        self.books = self.db.Book
        self.chapters = self.db.Chapter
        self.chat_storage = self.db.ChatStorage
        self.music = self.db.music

    def get_current_chapter_contents(self, user_id: str, book_id: str) -> Dict:
        """
        Get the current working chapter contents (workingFlag = True)
        Returns:
            {
                "chapter_info": {
                    "bookId": "book1",
                    "chapter_Num": "chapter_1",
                    "sumChapter": "요약",
                    "workingFlag": True,
                    "musicTitle": "노래 제목",
                    "composer": "작곡자"
                },
                "chat_contents": [
                    {"LLM_Model": "...", "User": "..."},
                    ...
                ]
            }
        """
        try:
            # Get current working chapter
            chapter = self.chapters.find_one({
                "userId": user_id,
                "bookId": book_id,
                "workingFlag": True
            })
            
            if not chapter:
                return None
                
            # Get chat contents
            chat = self.chat_storage.find_one({
                "chapter_Num": chapter["chapter_Num"]
            })
            
            # Remove MongoDB _id
            if chapter:
                chapter.pop("_id", None)
            
            return {
                "chapter_info": chapter,
                "chat_contents": chat.get("content", []) if chat else []
            }
        except Exception as e:
            raise Exception(f"Error getting current chapter contents: {str(e)}")

    def get_completed_chapters_contents(self, user_id: str, book_id: str) -> List[Dict]:
        """
        Get summary contents of all completed chapters (workingFlag = False)
        Returns only the summary information, not the full chat contents
        Returns:
            [
                {
                    "chapter_num": "chapter_1",
                    "summary": "챕터 요약 내용"
                },
                ...
            ]
        """
        try:
            # Get all completed chapters with only necessary fields
            chapters = list(self.chapters.find(
                {
                    "userId": user_id,
                    "bookId": book_id,
                    "workingFlag": False
                },
                {
                    "_id": 0,
                    "chapter_Num": 1,
                    "sumChapter": 1
                }
            ).sort("chapter_Num", 1))
            
            result = []
            for chapter in chapters:
                result.append({
                    "chapter_num": chapter.get("chapter_Num", ""),
                    "summary": chapter.get("sumChapter", "")
                })
            
            return result
        except Exception as e:
            raise Exception(f"Error getting completed chapters contents: {str(e)}")

    def get_user_data(self, user_id: str) -> Dict:
        """
        Get user's data including book and chapters
        """
        try:
            user = self.users.find_one({"userId": user_id})
            if not user:
                return None
            
            book = self.books.find_one({"userId": user_id})
            if not book:
                return {
                    "user": user,
                    "book": None,
                    "chapters": []
                }
            
            chapters = list(self.chapters.find(
                {"userId": user_id, "bookId": book["bookId"]},
                {"_id": 0}
            ).sort("chapter_Num", 1))
            
            return {
                "user": user,
                "book": book,
                "chapters": chapters
            }
        except Exception as e:
            raise Exception(f"Error getting user data: {str(e)}")

    def update_chat_history(self, user_id: str, chapter_num: str, content: List[Dict]) -> bool:
        """
        Update chat history for a chapter
        
        2025-06-22 17:00
        chat_storage collection 구조 변경 및 접근 시 bookid 추가 해야함
        
        """
        try:
            self.chat_storage.update_one(
                {
                    "userId": user_id,
                    "chapter_Num": chapter_num
                },
                {
                    "$set": {
                        "content": content
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            raise Exception(f"Error updating chat history: {str(e)}")

    def get_chat_history(self, user_id: str, chapter_num: str) -> Optional[List[Dict]]:
        """
        Get chat history for a chapter
        
        2025-06-22 17:00
        chat_storage collection 구조 변경 및 접근 시 bookid 추가 해야함
        
        """
        try:
            chat = self.chat_storage.find_one({
                "userId": user_id,
                "chapter_Num": chapter_num
            })
            return chat.get("content", []) if chat else []
        except Exception as e:
            raise Exception(f"Error getting chat history: {str(e)}")

    def get_music_database_for_recommendation(self) -> List[Dict]:
        """
        Get music database formatted for the music recommender
        Converts mood_* fields to feature_vector format
        Returns music documents with:
        - songName: 음악 제목 (from 'name' field)
        - artist: 아티스트 (from 'artist' field)  
        - feature_vector: [acoustic, electronic, aggressive, relaxed, happy, sad, party] 7차원 벡터
        - mbid: music brainz id
        """
        try:
            # 모든 음악 데이터 조회
            music_list = list(self.music.find({}, {"_id": 0}))
            
            processed_music = []
            for music in music_list:
                # 기본 필드 매핑
                processed_music_item = {
                    "songName": music.get("name", "Unknown"),
                    "artist": music.get("artist", "Unknown"),
                    "mbid": music.get("mbid", ""),
                }
                
                # mood 필드들을 feature_vector로 변환
                # feature_names 순서: ['acoustic', 'electronic', 'aggressive', 'relaxed', 'happy', 'sad', 'party']
                feature_vector = [
                    music.get("mood_acoustic", 0.5),    # acoustic
                    music.get("mood_electronic", 0.5),  # electronic  
                    music.get("mood_aggressive", 0.5),  # aggressive
                    music.get("mood_relaxed", 0.5),     # relaxed
                    music.get("mood_happy", 0.5),       # happy
                    music.get("mood_sad", 0.5),         # sad
                    music.get("mood_party", 0.5)        # party
                ]
                
                processed_music_item["feature_vector"] = feature_vector
                processed_music_item["original_data"] = music  # 원본 데이터 보존
                
                processed_music.append(processed_music_item)
            
            return processed_music
        except Exception as e:
            raise Exception(f"Error getting music database for recommendation: {str(e)}")

    def update_chapter_music(self, user_id: str, book_id: str, chapter_num: str, music_data: Dict) -> bool:
        """
        Update music information for a chapter
        """
        try:
            update_data = {
                "musicTitle": music_data.get("musicTitle", ""),
                "composer": music_data.get("composer", ""),
            }

            result = self.chapters.update_one(
                {
                    "userId": user_id,
                    "bookId": book_id,
                    "chapter_Num": chapter_num
                },
                {
                    "$set": update_data
                }
            )
            return result.modified_count > 0
        except Exception as e:
            raise Exception(f"Error updating chapter music: {str(e)}")

    def get_chapter_contents(self, user_id: str, book_id: str, chapter_num: str) -> Dict:
        """
        Get specific chapter contents by user_id, book_id and chapter_num
        Returns:
            {
                "chapter_info": {
                    "bookId": "book1",
                    "chapter_Num": "chapter_1",
                    "sumChapter": "요약",
                    "workingFlag": True/False,
                    "musicTitle": "노래 제목",
                    "composer": "작곡자"
                },
                "chat_contents": [
                    {"LLM_Model": "...", "User": "..."},
                    ...
                ]
            }
        """
        try:
            # Get specific chapter
            chapter = self.chapters.find_one({
                "userId": user_id,
                "bookId": book_id,
                "chapter_Num": chapter_num
            })
            
            if not chapter:
                return None
                
            # Get chat contents
            chat = self.chat_storage.find_one({
                "chapter_Num": chapter_num
            })
            
            # Remove MongoDB _id
            if chapter:
                chapter.pop("_id", None)
            
            return {
                "chapter_info": chapter,
                "chat_contents": chat.get("content", []) if chat else []
            }
        except Exception as e:
            raise Exception(f"Error getting chapter contents: {str(e)}")

    def update_chapter_summary(self, user_id: str, book_id: str, chapter_num: str, summary: str) -> bool:
        """
        Update chapter summary (sumChapter field)
        """
        try:
            result = self.chapters.update_one(
                {
                    "userId": user_id,
                    "bookId": book_id,
                    "chapter_Num": chapter_num
                },
                {
                    "$set": {
                        "sumChapter": summary,
                    }
                }
            )
            return result.modified_count > 0
        except Exception as e:
            raise Exception(f"Error updating chapter summary: {str(e)}")

    def get_user_by_id(self, user_id: str) -> Dict:
        """
        Get user by user_id
        """
        try:
            user = self.users.find_one(
                {"userId": user_id},
                {"_id": 0}
            )
            return user
        except Exception as e:
            raise Exception(f"Error getting user: {str(e)}")

    def get_book_info(self, user_id: str, book_id: str) -> Optional[Dict]:
        """
        Book Collection에서 userId, bookId로 책 정보를 조회한다.
        반환값은 _id를 제외한 책의 모든 필드이다.
        """
        try:
            book = self.books.find_one({"userId": user_id, "bookId": book_id}, {"_id": 0})
            return book
        except Exception as e:
            raise Exception(f"Error getting book info: {str(e)}")