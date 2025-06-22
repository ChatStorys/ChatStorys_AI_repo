from typing import Dict
from dotenv import load_dotenv

# 프로젝트 내부 모듈 임포트 (Hugging Face 버전)
from .api.gpt_client import GPTClient
from .database.db_manager import DatabaseManager
from .utils.emotion_analyzer import EmotionAnalyzer
from .utils.music_recommender import MusicRecommender

# 환경변수 로드
load_dotenv()

class NovelProcessor:
    """소설 생성 및 처리를 담당하는 메인 클래스 (Hugging Face 모델 버전)"""
    
    def __init__(self, use_hf_model=True, hf_model_name=None):
        """
        소설 처리에 필요한 모든 클라이언트 초기화 (Hugging Face 버전)
        
        Args:
            use_hf_model (bool): Hugging Face 모델 사용 여부 (기본값: True)
            hf_model_name (str): Hugging Face 모델 이름 (기본값: "Jinuuuu/KoELECTRA_fine_tunning_emotion")
        """
        print("=== NovelProcessor 초기화 (Hugging Face 모델 버전) ===")
        
        self.gpt_client = GPTClient()
        self.db_manager = DatabaseManager()
        
        # Hugging Face 모델 사용
        if use_hf_model:
            print("Hugging Face KoELECTRA 모델을 사용합니다.")
            model_name = hf_model_name or "Jinuuuu/KoELECTRA_fine_tunning_emotion"
            self.emotion_analyzer = EmotionAnalyzer(
                model_name=model_name,
                use_local=False
            )
            self.music_recommender = MusicRecommender(
                db_manager=self.db_manager,
                use_hf_model=True,
                model_name=model_name
            )
        else:
            print("로컬 KoELECTRA 모델을 사용합니다.")
            self.emotion_analyzer = EmotionAnalyzer(use_local=True)
            self.music_recommender = MusicRecommender(
                db_manager=self.db_manager,
                use_hf_model=False
            )
        
        print("NovelProcessor 초기화 완료!")

    def generate_chapter(self, user_id: str, user_message: str, book_id: str) -> Dict:
        """
        소설 챕터 생성 함수 (Hugging Face 모델 버전)
        
        사용자의 입력을 받아 AI가 소설의 다음 내용을 생성합니다.
        이전 챕터들의 맥락과 현재 챕터의 채팅 히스토리를 모두 활용하여
        일관성 있는 스토리를 생성합니다.

        매개변수:
            user_id: 사용자의 고유 식별자
            user_message: 사용자가 입력한 메시지 (스토리 방향 지시)
            book_id: 현재 작성 중인 책의 고유 식별자

        반환값:
            성공 시: {"status": "success", "chapter": 챕터_데이터}
            실패 시: {"status": "error", "message": 오류_메시지}
        
        처리 과정:
            1. 사용자 존재 여부 및 책 정보 확인
            2. 이전 완료된 챕터들의 요약 정보 조회
            3. 현재 챕터의 채팅 히스토리 조회
            4. 컨텍스트 정보 구성 (이전 챕터 + 현재 진행상황)
            5. GPT를 이용한 소설 내용 생성
            6. 생성된 내용을 DB에 저장
            7. 결과 반환
        """
        try:
            print(f"[HF 모델] 챕터 생성 시작 - 사용자: {user_id}, 책: {book_id}")
            
            # 1. 사용자 존재 여부 확인
            user = self.db_manager.get_user_by_id(user_id)
            if not user:
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "사용자를 찾을 수 없습니다",
                    "prompt": None
                }
            
            # 2. 책 정보 확인 (사용자가 해당 책에 접근 권한이 있는지 확인)
            book_info = self.db_manager.get_book_info(user_id, book_id)
            if not book_info:
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "책 정보를 찾을 수 없습니다",
                    "prompt": None
                }
            
            # 3. 사용자가 해당 책의 소유자인지 확인
            if book_info.get('userId') != user_id:
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "해당 책에 대한 접근 권한이 없습니다",
                    "prompt": None
                }
            
            # 4. 이전 완료된 챕터들의 summary 조회 (book_id 직접 사용)
            # 성능 향상을 위해 전체 내용이 아닌 요약만 조회
            previous_chapters = self.db_manager.get_completed_chapters_contents(user_id=user_id, book_id=book_id)
            
            # 5. 현재 작업 중인 챕터의 채팅 히스토리 조회 (book_id 직접 사용)
            current_chapter = self.db_manager.get_current_chapter_contents(user_id=user_id, book_id=book_id)
            
            # 6. 현재 작업 중인 챕터가 없는 경우 처리
            if not current_chapter:
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "현재 작업 중인 챕터가 없습니다. 새 챕터를 시작해주세요.",
                    "prompt": None
                }
            
            chat_history = current_chapter.get('chat_contents', [])
            
            # 7. 채팅 히스토리를 연속된 텍스트로 변환
            continuous_text = ""
            for chat in chat_history:
                if "User" in chat:
                    continuous_text += f"사용자: {chat['User']}\n"
                if "LLM_Model" in chat:
                    continuous_text += f"AI: {chat['LLM_Model']}\n"
            
            # 8. GPT에 전달할 컨텍스트 정보 구성
            # 이전 완료된 챕터들의 summary만 사용
            context = {
                "previous_chapters": "\n".join([
                    f"챕터 {ch['chapter_num']}: {ch['summary']}" for ch in previous_chapters if ch['summary']
                ]),
                "current_chapter": continuous_text,
                "book_title": book_info.get('title', ''),
                "book_genre": book_info.get('genre', ''),
            }
            
            # 9. 현재 챕터 번호 확인
            chapter_num = current_chapter.get('chapter_info', {}).get('chapter_Num', '')
            
            print(f"[HF 모델] 챕터 {chapter_num} GPT 생성 요청")
            
            # 10. GPT를 이용하여 소설 내용 생성 (개선된 chat_session 사용)
            content = self.gpt_client.chat_session(
                chapter_num=chapter_num,
                context=context,
                user_message=user_message,
                messages=chat_history  # 채팅 히스토리 전체 전달
            )
            
            # 11. 채팅 히스토리 업데이트
            updated_chat_history = current_chapter.get('chat_contents', [])
            updated_chat_history.append({"LLM_Model": content, "User": user_message})
            self.db_manager.update_chat_history(user_id, chapter_num, updated_chat_history)
            
            print(f"[HF 모델] 챕터 생성 완료 - 길이: {len(content)}자")
            
            return {
                "status": "success",
                "code": 200,
                "message": "소설 저장 완료",
                "prompt": content
            }
            
        except Exception as e:
            print(f"[HF 모델] 챕터 생성 오류: {str(e)}")
            return {
                "status": "fail",
                "code": 500,
                "message": "소설 저장 중 오류가 발생했습니다",
                "prompt": None
            }

    def finish_chapter_and_recommend_music(self, user_id: str, book_id: str) -> Dict:
        """
        챕터 완료 및 음악 추천 함수 (Hugging Face 모델 버전)
        현재 작업 중인 챕터(workingFlag=True)의 내용을 요약하고 DB에 저장한 후,
        Algorithm 1을 사용하여 적절한 음악을 추천합니다.
        모든 처리가 완료되면 workingFlag를 False로 변경합니다.
        """
        try:
            print(f"[HF 모델] 챕터 완료 및 음악 추천 시작 - 사용자: {user_id}, 책: {book_id}")
            # 1. user_id, book_id로 현재 작업 중인 챕터 조회
            current_chapter = self.db_manager.get_current_chapter_contents(user_id=user_id, book_id=book_id)
            if not current_chapter:
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "텍스트 추천 실패 (CODE 500)"
                }
            
            chapter_num = current_chapter.get('chapter_info', {}).get('chapter_Num', '')
            if not chapter_num:
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "텍스트 추천 실패 (CODE 500)"
                }
            
            # 2. 챕터의 채팅 내용을 하나의 연속된 텍스트로 변환
            chat_contents = current_chapter.get('chat_contents', [])
            chapter_content_text = ""
            
            for chat in chat_contents:
                if "User" in chat:
                    chapter_content_text += f"사용자: {chat['User']}\n"
                if "LLM_Model" in chat:
                    chapter_content_text += f"AI: {chat['LLM_Model']}\n"
            
            if not chapter_content_text.strip():
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "텍스트 추천 실패 (CODE 500)"
                }
            
            # 3. GPT를 통한 챕터 요약 생성
            try:
                chapter_summary = self.gpt_client.summarize_chapter(
                    content=chapter_content_text,
                    chapter_num=chapter_num
                )
                
            except Exception as e:
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "텍스트 추천 실패 (CODE 500)"
                }
            
            # 4. 챕터 요약 정보 업데이트
            try:
                self.db_manager.update_chapter_summary(
                    user_id=user_id,
                    book_id=book_id,
                    chapter_num=chapter_num,
                    summary=chapter_summary
                )
                
            except Exception as e:
                return {
                    "status": "fail",
                    "code": 500,
                    "message": "텍스트 추천 실패 (CODE 500)"
                }
            
            # 5. Algorithm 1을 사용한 음악 추천 (Hugging Face 모델)
            try:
                # Algorithm 1: recommend_music에서 클래스 내부의 db_manager를 통해 음악 데이터베이스 조회 및 추천 처리
                # 반드시 전체 챕터 내용을 기반으로 음악을 추천한다.
                recommendations = self.music_recommender.recommend_music(
                    userID=user_id,
                    novelContents=chapter_content_text,  # 반드시 전체 챕터 내용 사용
                    musicDB=None,
                    N=1
                )
                
                if recommendations and len(recommendations) > 0:
                    # 가장 유사도가 높은 음악 선택
                    selected_music = recommendations[0]
                    
                    # 6. 챕터에 음악 정보 저장 및 완료 처리
                    try:
                        self.db_manager.update_chapter_music(
                            user_id=user_id,
                            book_id=book_id,
                            chapter_num=chapter_num,
                            music_data={
                                "musicTitle": selected_music.get('songName'),
                                "composer": selected_music.get('artist')
                            }
                        )
                        
                        # 챕터 완료 처리 (workingFlag를 False로 변경)
                        # server에서 처리
                        # self.db_manager.complete_chapter(book_id, chapter_num)
                        
                    except Exception as e:
                        return {
                            "status": "fail",
                            "code": 500,
                            "message": "텍스트 추천 실패 (CODE 500)"
                        }
                    
                    return {
                        "status": "success",
                        "code": 200,
                        "summary": chapter_summary,
                        "recommended_music": [{
                            "title": selected_music.get('songName'),
                            "artist": selected_music.get('artist')
                        }]
                    }
                else:
                    # 추천 결과가 없는 경우
                    return {
                        "status": "fail",
                        "code": 500,
                        "message": "텍스트 추천 실패 (CODE 500)"
                    }
                    
            except Exception as e:
                print(f"Algorithm 1 음악 추천 오류: {str(e)}")
                
                # 대안: 기존 방식으로 폴백
                try:
                    # KoELECTRA 모델을 이용한 감정 분석 (폴백) - 전체 챕터 내용 사용
                    emotions = self.emotion_analyzer.analyze_emotions(chapter_content_text)
                    dominant_emotion = emotions.get('dominant_emotion', 'happy')
                    
                    # 주요 감정에 기반한 음악 추천 (기존 방식)
                    music_list = self.db_manager.get_music_by_emotion(dominant_emotion)
                    
                    if music_list:
                        selected_music = music_list[0]
                        
                        # 챕터에 음악 정보 저장하고 완료 처리
                        self.db_manager.update_chapter_music(
                            user_id=user_id,
                            book_id=book_id,
                            chapter_num=chapter_num,
                            music_data={
                                "musicTitle": selected_music.get('musicTitle'),
                                "composer": selected_music.get('composer'),
                            }
                        )
                        
                        self.db_manager.complete_chapter(book_id, chapter_num)
                        
                        return {
                            "status": "success",
                            "code": 200,
                            "summary": chapter_summary,
                            "recommended_music": [{
                                "title": selected_music.get('musicTitle'),
                                "artist": selected_music.get('composer')
                            }]
                        }
                    else:
                        # 음악을 찾을 수 없는 경우에도 요약은 저장하고 완료 처리
                        self.db_manager.complete_chapter(book_id, chapter_num)
                        
                        return {
                            "status": "fail",
                            "code": 500,
                            "message": "텍스트 추천 실패 (CODE 500)"
                        }
                        
                except Exception as fallback_error:
                    return {
                        "status": "fail",
                        "code": 500,
                        "message": "텍스트 추천 실패 (CODE 500)"
                    }
            
        except Exception as e:
            return {
                "status": "fail",
                "code": 500,
                "message": "텍스트 추천 실패 (CODE 500)"
            }

# 전역 소설 처리 인스턴스 (지연 초기화) - Hugging Face 버전
novel_processor = None

def get_novel_processor():
    """NovelProcessor 인스턴스 반환 (지연 초기화)"""
    global novel_processor
    if novel_processor is None:
        novel_processor = NovelProcessor()
    return novel_processor

def handle_story_continue(user_id: str, user_message: str, book_id: str) -> Dict:
    """
    소설 계속 쓰기 요청 처리 함수 (Hugging Face 모델 버전)
    
    외부 서버에서 호출할 수 있는 함수입니다.
    /story/continue 엔드포인트의 요청을 처리합니다.
    
    매개변수:
        user_id: 사용자 ID
        user_message: 사용자 메시지
        book_id: 책 ID
        
    반환값:
        처리 결과 딕셔너리
    """
    try:
        result = get_novel_processor().generate_chapter(
            user_id=user_id,
            user_message=user_message,
            book_id=book_id
        )
        return result
        
    except Exception as e:
        return {
            "status": "fail",
            "code": 500,
            "message": "소설 저장 중 오류가 발생했습니다",
            "prompt": None
        }

def handle_chapter_summary_with_music(user_id: str, book_id: str) -> Dict:
    """
    챕터 요약 및 음악 추천 요청 처리 함수 (Hugging Face 모델 버전)
    
    외부 서버에서 호출할 수 있는 함수입니다.
    /story/chapter/summary_with_music 엔드포인트의 요청을 처리합니다.
    현재 작업 중인 챕터(workingFlag=True)를 찾아 요약을 생성하고 음악을 추천합니다.
    
    매개변수:
        user_id: 사용자 ID
        book_id: 책 ID
        
    반환값:
        처리 결과 딕셔너리
    """
    try:
        result = get_novel_processor().finish_chapter_and_recommend_music(
            user_id=user_id,
            book_id=book_id
        )
        return result
        
    except Exception as e:
        return {
            "status": "fail",
            "code": 500,
            "message": "텍스트 추천 실패 (CODE 500)"
        }