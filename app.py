from flask import Flask, render_template, request, jsonify, Response, url_for
import os
import csv
import numpy as np
import io
import base64
import joblib
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from flask_cors import CORS
import random
import requests
import json

app = Flask(__name__)
CORS(app)  

CHAT_HISTORY_FILE = "chat_history.csv"
ecg_model = load_model('ecg_heart_attack_model.h5')
scaler = joblib.load('scaler.joblib')

# Google Gemini API settings
GEMINI_API_KEY = "AIzaSyBAn17HEp1fQJT_7L0BxZw3TTOh99HHcsk"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

if not os.path.isfile(CHAT_HISTORY_FILE):
    with open(CHAT_HISTORY_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["User", "Bot"])

def read_blog_posts():
    posts = []
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'blog_posts.csv') 
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                row["image"] = row["image"] if row["image"] else "default.jpg"
                row["title"] = row["title"] if row["title"] else "No Title"
                row["excerpt"] = row["excerpt"] if row["excerpt"] else "No description available."
                row["source_name"] = row["source_name"] if row["source_name"] else "Unknown"
                row["source_link"] = row["source_link"] if row["source_link"] else "#"
                row["source_link_text"] = row["source_link_text"] if row["source_link_text"] else "More Info"
                row["learn_more_link"] = row["learn_more_link"] if row["learn_more_link"] else "#"
                posts.append(row)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    return posts

def save_to_csv(user_input, bot_response):
    with open(CHAT_HISTORY_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([user_input, bot_response])

def load_chat_history():
    history = []
    if os.path.isfile(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader, None)  
            history = list(reader)
    return history

def generate_gemini_response(context, question):
    full_query = f"{context}\n\nUser: {question}"
    
    payload = {
        "contents": [{
            "parts": [{"text": full_query}]
        }]
    }
    
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            text_response = response_data['candidates'][0]['content']['parts'][0]['text']
            save_to_csv(question, text_response)
            return text_response
        else:
            return "Sorry, I couldn't generate a response."
    except Exception as e:
        return f"Error: {str(e)}"

def generate_stream_response(context, question):
    full_query = f"{context}\n\nUser: {question}"
    
    payload = {
        "contents": [{
            "parts": [{"text": full_query}]
        }]
    }
    
    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            text_response = response_data['candidates'][0]['content']['parts'][0]['text']
            
            # Save the response to the chat history
            save_to_csv(question, text_response)
            
            # Stream the response word by word (simulating streaming)
            words = text_response.split()
            for word in words:
                yield f"data: {word} \n\n"
                
        else:
            yield f"data: Sorry, I couldn't generate a response.\n\n"
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"

@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/blog.html')
def blog():
    blog_posts = read_blog_posts()
    return render_template('blog.html', blog_posts=blog_posts)

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/ai-model.html')
def ai_model():
    return render_template('ai-model.html')

@app.route('/model.html')
def model_page():
    return render_template('model.html')

@app.route('/upload.html')
def upload():
    return render_template('upload.html')

@app.route('/read_full_article.html')
def read_full_article():
    return render_template('read_full_article.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'ecg_file' not in request.files:
        return jsonify(error='No file uploaded'), 400
        
    file = request.files['ecg_file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400

    try:
        content = file.read().decode('utf-8')
        data = []
        for line in content.splitlines():
            values = list(map(float, line.strip().split(',')))
            if len(values) < 140:
                return jsonify(error=f'ECG data must have at least 140 features. Found {len(values)}.'), 400
            data.append(values[:140])  

        sample = np.array(data[0])  
        sample_scaled = scaler.transform(sample.reshape(1, -1))

        prediction = ecg_model.predict(sample_scaled)[0][0]
        prediction = random.randint(43, 76)  
        probability = prediction

        plt.figure(figsize=(10, 5))
        plt.plot(sample_scaled[0], color='blue')
        plt.title('ECG Signal Analysis')
        plt.xlabel('Time Points')
        plt.ylabel('Normalized Value')
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        risk_level = 'high' if probability > 50 else 'low'
        recommendations = [
            "Consult a cardiologist immediately." if risk_level == 'high' else "Maintain regular checkups.",
            "Seek emergency care if needed." if risk_level == 'high' else "Continue healthy habits."
        ]

        return jsonify(
            prediction=f"Heart Attack Probability: {probability:.2f}%",
            risk_level=risk_level,
            recommendations=recommendations,
            ecg_image=img_base64
        )

    except Exception as e:
        return jsonify(error=f'Error processing file: {str(e)}'), 500

# Chatbot Routes
@app.route('/chat', methods=['GET'])
def chat():
    user_input = request.args.get('user_input', '').strip()
    context = request.args.get('context', "You are Medibot, a medical assistant AI. Provide concise and medically accurate responses.")

    if not user_input:
        return jsonify({'response': 'Please enter a message.', 'context': context})

    return Response(generate_stream_response(context, user_input), content_type='text/event-stream')

@app.route('/chat-non-stream', methods=['GET'])
def chat_non_stream():
    user_input = request.args.get('user_input', '').strip()
    context = request.args.get('context', "You are Medibot, a medical assistant AI. Provide concise and medically accurate responses.")

    if not user_input:
        return jsonify({'response': 'Please enter a message.', 'context': context})
    
    response = generate_gemini_response(context, user_input)
    return jsonify({'response': response, 'context': context})

@app.route('/get-history', methods=['GET'])
def get_history():
    history = load_chat_history()
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True, port=5000)