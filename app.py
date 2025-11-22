# backend/app.py

from flask import Flask, render_template, session
from flask_pymongo import PyMongo
import os
from dotenv import load_dotenv

load_dotenv()

# Імпортуємо з поточної папки (backend)
from config import MONGO_URI, SECRET_KEY
from routes.auth import auth_bp
from routes.planner import planner_bp
from routes.ai import ai_bp

app = Flask(__name__, template_folder='templates', static_folder='static')

app.config['SECRET_KEY'] = SECRET_KEY
app.config['MONGO_URI'] = MONGO_URI

try:
    mongo = PyMongo(app)
    app.config['db'] = mongo.db
    print("✓ MongoDB configured successfully")
except Exception as e:
    print(f"✗ MongoDB configuration error: {e}")
    mongo = None

# Реєстрація blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(planner_bp)
app.register_blueprint(ai_bp)

@app.route('/')
def index():
    # Варіант 1: Завжди показувати index.html
    return render_template('index.html')
    
    # АБО Варіант 2: Перенаправляти на планер
    # if 'user_id' in session:
    #     return redirect(url_for('planner.calendar_view'))
    # return render_template('index.html')

@app.route('/health')
def health():
    try:
        if mongo and app.config.get('db'):
            app.config['db'].command('ping')
            return {'status': 'healthy', 'db': 'connected'}, 200
        else:
            return {'status': 'unhealthy', 'db': 'not_initialized'}, 500
    except Exception as e:
        print(f"Health check error: {e}")
        return {'status': 'unhealthy', 'error': str(e)}, 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    print(f"Server error: {e}")
    return {'error': 'Internal server error'}, 500

if __name__ == "__main__":
    debug_mode = os.getenv('FLASK_ENV', 'development') == 'development'
    port = int(os.getenv('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)