# backend/routes/ai.py

from flask import Blueprint, request, jsonify, session, current_app
from datetime import datetime, timedelta
from bson.objectid import ObjectId
import os
from dotenv import load_dotenv
from pathlib import Path
import requests
import json
import re
import random
import time
import sys

# Завантажуємо .env
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

ai_bp = Blueprint('ai', __name__, url_prefix='/api/ai')

HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

# --- ОНОВЛЕНИЙ URL ДЛЯ HUGGINGFACE API ---
HF_MODEL_URL = "https://router.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Альтернативний URL, якщо основний не працює
HF_MODEL_URL_ALT = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

class HuggingFaceAnalyzer:
    def query_hf(self, payload):
        """
        Відправляє запит з логікою очікування та автоматичним переключенням URL.
        """
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        # Список URL для спроб (новий перший)
        urls_to_try = [HF_MODEL_URL, HF_MODEL_URL_ALT]
        
        for url in urls_to_try:
            for i in range(3):  # 3 спроби для кожного URL
                try:
                    print(f"🔄 Спроба {i+1} для URL: {url}")
                    sys.stdout.flush()
                    
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    
                    # Якщо модель вантажиться (503)
                    if response.status_code == 503:
                        try:
                            data = response.json()
                            wait_time = data.get('estimated_time', 10)
                            print(f"⏳ DeepSeek завантажується... Чекаємо {wait_time} сек.")
                            sys.stdout.flush()
                            time.sleep(wait_time)
                            continue
                        except:
                            time.sleep(5)
                            continue

                    # Якщо успіх
                    if response.status_code == 200:
                        return response.json()

                    # Якщо помилка клієнта (4xx) - не повторюємо
                    if 400 <= response.status_code < 500:
                        print(f"❌ Client Error ({response.status_code}): {response.text}")
                        # Переходимо до наступного URL
                        break

                    # Якщо серверна помилка (5xx) - повторюємо
                    print(f"❌ Server Error ({response.status_code}): {response.text}")
                    time.sleep(2)
                    continue

                except requests.exceptions.Timeout:
                    print(f"⏰ Timeout для {url}, спроба {i+1}")
                    time.sleep(3)
                    continue
                except requests.exceptions.ConnectionError as e:
                    print(f"🔌 Connection Error для {url}: {e}")
                    sleep(2)
                    continue
                except Exception as e:
                    print(f"❌ Unexpected Error для {url}: {e}")
                    time.sleep(2)
                    continue
        
        return {"error": "All API endpoints failed"}

    def get_dynamic_persona(self, mood_rating):
        """
        Ролі для DeepSeek.
        """
        if mood_rating <= 4:
            return {
                "role": "Турботливий друг",
                "style": "Теплі слова, емпатія, підтримка.",
                "example": "Ох, тримайся! Ти все одно молодець."
            }
        elif 5 <= mood_rating <= 7:
            return {
                "role": "Веселий колега",
                "style": "Легкий гумор, іронія, позитив.",
                "example": "Нормальний день, жити можна!"
            }
        else:
            return {
                "role": "Енергійний фанат",
                "style": "Захват, енергія, радість!",
                "example": "Ти просто космос! Це був неймовірний день!"
            }

    def parse_json_safely(self, text, default_score):
        try:
            # 1. ОЧИСТКА ВІД "ДУМОК" (DeepSeek-R1)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            
            # 2. Пошук JSON
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx == -1 or end_idx == -1: 
                return None
            
            json_str = text[start_idx:end_idx+1]
            json_str = json_str.replace('\n', ' ')
            
            try:
                return json.loads(json_str)
            except:
                # Спроба виправити поширені проблеми з JSON
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r',\s*}', '}', json_str)  # Видаляємо зайві коми
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)
        except:
            return None

    def analyze_day(self, text, completed_tasks, total_tasks, user_mood_rating):
        # Fallback відповідь на випадок помилки
        fallback_response = {
            "mood_score": user_mood_rating,
            "summary": self.get_fallback_summary(user_mood_rating, completed_tasks, total_tasks)
        }
        
        # Перевірка на наявність API ключа
        if not HF_API_KEY:
            print("⚠️ HUGGINGFACE_API_KEY не знайдено")
            return fallback_response

        persona = self.get_dynamic_persona(user_mood_rating)

        # Покращений промпт
        prompt = f"""<|user|>
Roleplay: You are a {persona['role']}.
Task: Analyze the user's diary and return a JSON summary in UKRAINIAN.
Style Guide: {persona['style']}
Example Tone: "{persona['example']}"

Input Data:
- Mood Rating: {user_mood_rating}/10
- Tasks Completed: {completed_tasks}/{total_tasks}
- Diary Text: "{text}"

Format Requirement:
Output ONLY the JSON object. Do not output reasoning.

JSON Structure:
{{
    "mood_score": {user_mood_rating},
    "summary": "Твій текст українською тут..."
}}
<|end|>
<|assistant|>
"""

        try:
            print(f"🚀 Запит до DeepSeek-R1 (1.5B)... (Роль: {persona['role']})")
            sys.stdout.flush()
            
            output = self.query_hf({
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "return_full_text": False,
                    "temperature": 0.6,
                    "do_sample": True
                }
            })

            if isinstance(output, dict) and 'error' in output:
                print(f"❌ API Error: {output['error']}")
                return fallback_response

            content = ""
            if isinstance(output, list) and len(output) > 0:
                content = output[0].get('generated_text', '')
            elif isinstance(output, dict):
                content = output.get('generated_text', '')

            if not content:
                print("❌ Порожня відповідь від API")
                return fallback_response

            print(f"✅ Відповідь отримана: {content[:100]}...") 
            sys.stdout.flush()

            result = self.parse_json_safely(content, user_mood_rating)
            
            if result:
                if 'summary' not in result: 
                    result['summary'] = fallback_response['summary']
                if 'mood_score' not in result: 
                    result['mood_score'] = user_mood_rating
                return result
            
            # Fallback - якщо JSON не розпарсено, використовуємо текст як є
            clean_text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            clean_text = clean_text.replace('```json', '').replace('```', '').strip()
            
            if len(clean_text) > 10:
                return {
                    "mood_score": user_mood_rating, 
                    "summary": clean_text[:300] + "..." if len(clean_text) > 300 else clean_text
                }

            return fallback_response

        except Exception as e:
            print(f"🔥 CRITICAL ERROR: {e}")
            sys.stdout.flush()
            return fallback_response

    def get_fallback_summary(self, mood_rating, completed_tasks, total_tasks):
        """Fallback відповіді на випадок помилки API"""
        if mood_rating <= 4:
            messages = [
                "Бачу, що день був непростим. Пам'ятай - кожна складна ситуація робить нас сильнішими.",
                "Іноді бувають такі дні. Відпочинь і завтра все обов'язково буде краще!",
                "Ти впорався! Навіть у складні дні ти знаходиш сили рухатись далі."
            ]
        elif 5 <= mood_rating <= 7:
            messages = [
                "Непоганий день! Маленькі кроки теж ведуть до великих цілей.",
                "Стабільний день - це теж досягнення. Ти на правильному шляху!",
                f"Чудово! Виконано {completed_tasks} з {total_tasks} завдань. Це гідна робота!"
            ]
        else:
            messages = [
                "Вражаюче! Твоя енергія та ентузіазм надихають!",
                "Неймовірний день! Так тримати! Ти демонструєш чудові результати!",
                f"Епічно! {completed_tasks} завершених завдань - це показник твоєї продуктивності!"
            ]
        
        return random.choice(messages)

    def calculate_productivity(self, completed_tasks, total_tasks, mood_score):
        if total_tasks == 0: 
            return int(mood_score * 10)
        completion_rate = (completed_tasks / total_tasks) * 100
        productivity = (completion_rate * 0.6) + (mood_score * 10 * 0.4)
        return int(min(100, max(0, productivity)))

    def map_score_to_label(self, score):
        """Повертає текстову мітку настрою"""
        mood_mapping = {
            1: "😢 Дуже сумний", 2: "😢 Сумний", 3: "😕 Розчарований",
            4: "😕 Втомлений", 5: "😐 Нормальний", 6: "🙂 Непоганий",
            7: "🙂 Добрий", 8: "😊 Чудовий", 9: "🤩 Енергійний", 10: "🔥 Неймовірний"
        }
        score = max(1, min(10, int(round(score))))
        return mood_mapping.get(score, "😐 Нормальний")

    def map_score_to_status_text(self, score):
        """Повертає текст статусу для відображення під стікером"""
        if score <= 3:
            return "Тримайся!"
        elif score <= 5:
            return "Все буде добре"
        elif score <= 7:
            return "Непогано!"
        else:
            return "Чудово!"

# ===== API Endpoints =====

@ai_bp.route('/analyze-entry', methods=['POST'])
def analyze_entry():
    if 'user_id' not in session: 
        return jsonify({'status': 'error', 'message': 'Необхідно увійти в систему'}), 401
    
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    user_mood_input = int(data.get('mood_rating', 5))
    date_str = data.get('date')
    
    if not text or not date_str: 
        return jsonify({'status': 'error', 'message': 'Відсутній текст або дата'}), 400
    
    try:
        db = current_app.config['db']
        user_id = ObjectId(session['user_id'])
        entry_date = datetime.strptime(date_str, '%Y-%m-%d')
        next_day = entry_date + timedelta(days=1)
        
        # Отримуємо події за день
        events = list(db.events.find({"user_id": user_id, "start_time": {"$gte": entry_date, "$lt": next_day}}))
        total_events = len(events)
        completed_events = len([e for e in events if e.get('is_completed', False)])
        
        # Аналіз через AI
        analyzer = HuggingFaceAnalyzer()
        ai_result = analyzer.analyze_day(text, completed_events, total_events, user_mood_input)
        
        ai_mood_score = ai_result.get('mood_score', user_mood_input)
        summary = ai_result.get('summary', '...')
        mood_label = analyzer.map_score_to_label(ai_mood_score)
        status_text = analyzer.map_score_to_status_text(ai_mood_score)  # Новий метод для тексту статусу
        productivity_score = analyzer.calculate_productivity(completed_events, total_events, ai_mood_score)
        
        # Зберігаємо запис користувача
        db.day_entries.update_one(
            {"user_id": user_id, "entry_date": entry_date},
            {"$set": {
                "user_description": text, 
                "user_mood_rating": user_mood_input, 
                "submitted_at": datetime.utcnow()
            }},
            upsert=True
        )
        
        # Зберігаємо аналіз AI
        ai_doc = {
            "user_id": user_id, 
            "date": entry_date, 
            "ai_mood_score": ai_mood_score,
            "ai_mood_label": mood_label, 
            "ai_status_text": status_text,  # Зберігаємо також текст статусу
            "ai_summary": summary, 
            "productivity_score": productivity_score,
            "completed_tasks": completed_events,
            "total_tasks": total_events,
            "created_at": datetime.utcnow()
        }
        db.ai_analyses.insert_one(ai_doc)
        
        return jsonify({
            'status': 'success', 
            'mood': mood_label, 
            'status_text': status_text,  # Додаємо текст статусу у відповідь
            'summary': summary,
            'productivity': productivity_score, 
            'score': productivity_score,
            'mood_rating': ai_mood_score, 
            'completed_tasks': completed_events, 
            'total_tasks': total_events
        }), 200

    except Exception as e:
        print(f"Analyze Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@ai_bp.route('/day-stats/<date_str>', methods=['GET'])
def get_day_stats(date_str):
    if 'user_id' not in session: 
        return jsonify({'status': 'error', 'message': 'Необхідно увійти в систему'}), 401
    try:
        db = current_app.config['db']
        user_id = ObjectId(session['user_id'])
        entry_date = datetime.strptime(date_str, '%Y-%m-%d')
        next_day = entry_date + timedelta(days=1)
        events = list(db.events.find({"user_id": user_id, "start_time": {"$gte": entry_date, "$lt": next_day}}))
        time_planned = sum([30 for _ in events])
        return jsonify({
            'total_events': len(events), 
            'completed_events': len([e for e in events if e.get('is_completed')]), 
            'time_planned_minutes': time_planned
        }), 200
    except Exception as e:
        print(f"Day Stats Error: {e}")
        return jsonify({'total_events': 0, 'completed_events': 0, 'time_planned_minutes': 0}), 200

@ai_bp.route('/chart-data', methods=['GET'])
def get_chart_data():
    if 'user_id' not in session: 
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        db = current_app.config['db']
        user_id = ObjectId(session['user_id'])
        cursor = db.ai_analyses.find({"user_id": user_id}).sort("date", 1).limit(7)
        data = list(cursor)
        
        return jsonify({
            "labels": [e['date'].strftime('%d.%m') for e in data],
            "moods": [e.get('ai_mood_score', 0) for e in data],
            "productivity": [e.get('productivity_score', 0) for e in data]
        })
    except Exception as e:
        print(f"Chart Data Error: {e}")
        return jsonify({"labels": [], "moods": [], "productivity": []})