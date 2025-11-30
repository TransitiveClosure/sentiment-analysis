import json
import os
import pickle
import re
from functools import wraps

import joblib
import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, Response, redirect, url_for, session
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, classification_report, accuracy_score

matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = 'super_secret_hackathon_key'

# Constants
DEFAULT_DATA_PATH = os.path.join('data', 'mock_train.csv')
USERS_FILE = os.path.join('data', 'users.json')
MODEL_PATH = 'ridge_max_model.pkl'


# User Management Helpers
def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
            return {}
    return {}


def save_users(users):
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Error saving users: {e}")
        return False


# Login Required Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# 1. Очистка текста
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    allowed = r"a-zа-яё0-9!?.,:;()\-\s"
    text = re.sub(fr"[^{allowed}]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 2. TextStatsExtractor
class TextStatsExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        stats = np.array([
            [
                len(x),
                x.count('!'),
                x.count('?'),
                x.count('.'),
                x.count('URL'),
                x.count('USER'),
            ]
            for x in X
        ])
        return stats


# Load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully with pickle")
    except Exception as e:
        print(f"Pickle load failed: {e}. Trying joblib...")
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully with joblib")
        except Exception as e2:
            print(f"Error loading model with joblib: {e2}")


def get_user_data_path():
    """Get the path to the current user's data file."""
    if 'user' in session:
        username = session['user']['username']
        # Sanitize username to prevent path traversal
        safe_username = "".join([c for c in username if c.isalpha() or c.isdigit() or c == '_']).rstrip()
        return os.path.join('data', f'{safe_username}_data.csv')
    return DEFAULT_DATA_PATH


def load_data():
    """Load data for the current user. Falls back to default data if user has no data."""
    user_path = get_user_data_path()

    # Determine which file to load
    path_to_load = user_path if os.path.exists(user_path) else DEFAULT_DATA_PATH

    if os.path.exists(path_to_load):
        try:
            df = pd.read_csv(path_to_load, engine='python', on_bad_lines='skip')
            # Map labels to human readable
            label_map = {0: 'Отрицательный', 1: 'Нейтральный', 2: 'Положительный'}
            df['label_text'] = df['label'].map(label_map)
            # Ensure ID is present, if not create it
            if 'ID' not in df.columns:
                df['ID'] = range(1, len(df) + 1)
            return df
        except Exception as e:
            print(f"Error reading CSV from {path_to_load}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def save_data(df):
    """Save data to the current user's specific file."""
    user_path = get_user_data_path()
    try:
        # Drop label_text before saving as it's a computed column
        if 'label_text' in df.columns:
            df = df.drop(columns=['label_text'])
        df.to_csv(user_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving CSV to {user_path}: {e}")
        return False


def apply_filters(df, source, sentiment, search):
    if source and source != 'all':
        df = df[df['src'] == source]

    if sentiment and sentiment != 'all':
        try:
            sentiment_val = int(sentiment)
            df = df[df['label'] == sentiment_val]
        except ValueError:
            pass

    if search:
        # Case-insensitive search in text
        df = df[df['text'].str.contains(search, case=False, na=False)]
    return df


def calculate_stats(df):
    total_reviews = len(df)
    sentiment_counts = df['label'].value_counts().to_dict()
    source_counts = df['src'].value_counts().to_dict()

    # Prepare data for Stacked Bar Chart (Sentiment by Source)
    top_sources = df['src'].value_counts().head(10).index.tolist()

    sentiment_by_source = {
        'labels': top_sources,
        'datasets': {
            0: [],  # Negative
            1: [],  # Neutral
            2: []  # Positive
        }
    }

    for src in top_sources:
        src_df = df[df['src'] == src]
        counts = src_df['label'].value_counts().to_dict()
        sentiment_by_source['datasets'][0].append(counts.get(0, 0))
        sentiment_by_source['datasets'][1].append(counts.get(1, 0))
        sentiment_by_source['datasets'][2].append(counts.get(2, 0))

    return {
        'sentiment': {
            'labels': ['Отрицательный', 'Нейтральный', 'Положительный'],
            'data': [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0), sentiment_counts.get(2, 0)]
        },
        'sources': {
            'labels': list(source_counts.keys()),
            'data': list(source_counts.values())
        },
        'sentiment_by_source': sentiment_by_source,
        'stats': {
            'total': total_reviews,
            'positive': sentiment_counts.get(2, 0),
            'neutral': sentiment_counts.get(1, 0),
            'negative': sentiment_counts.get(0, 0)
        }
    }


@app.route('/')
@login_required
def index():
    df = load_data()
    chart_data = calculate_stats(df)

    # Convert dataframe to dict for table - ONLY FIRST 10
    reviews = df.head(10).to_dict('records')

    return render_template('index.html',
                           reviews=reviews,
                           stats=chart_data['stats'],
                           chart_data=chart_data)


@app.route('/api/reviews')
@login_required
def get_reviews():
    offset = int(request.args.get('offset', 0))
    limit = int(request.args.get('limit', 50))
    source = request.args.get('source')
    sentiment = request.args.get('sentiment')
    search = request.args.get('search')

    df = load_data()
    df = apply_filters(df, source, sentiment, search)

    # Slice data
    reviews_chunk = df.iloc[offset:offset + limit].to_dict('records')

    return jsonify(reviews_chunk)


@app.route('/api/stats')
@login_required
def get_stats():
    source = request.args.get('source')
    sentiment = request.args.get('sentiment')
    search = request.args.get('search')

    df = load_data()
    df = apply_filters(df, source, sentiment, search)

    chart_data = calculate_stats(df)

    return jsonify(chart_data)


@app.route('/api/reviews/update', methods=['POST'])
@login_required
def update_review():
    data = request.json
    review_id = data.get('id')
    new_text = data.get('text')
    new_label = data.get('label')

    if review_id is None:
        return jsonify({'status': 'error', 'message': 'Missing ID'}), 400

    df = load_data()

    # Find index of review with this ID
    # Assuming ID is unique. If ID was generated on load, this might be tricky if CSV changes.
    # But for now we rely on the ID column we added/ensured.
    mask = df['ID'] == int(review_id)

    if not mask.any():
        return jsonify({'status': 'error', 'message': 'Review not found'}), 404

    if new_text:
        df.loc[mask, 'text'] = new_text
    if new_label is not None:
        df.loc[mask, 'label'] = int(new_label)

    if save_data(df):
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to save'}), 500


@app.route('/api/export')
@login_required
def export_data():
    source = request.args.get('source')
    sentiment = request.args.get('sentiment')
    search = request.args.get('search')

    df = load_data()
    df = apply_filters(df, source, sentiment, search)

    # Drop label_text if exists
    if 'label_text' in df.columns:
        df = df.drop(columns=['label_text'])

    csv_data = df.to_csv(index=False)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=reviews_export.csv"}
    )


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            try:
                # Read CSV
                df = pd.read_csv(file)

                # Ensure 'text' column exists
                if 'text' not in df.columns:
                    print("No 'text' column found")
                    return "Error: CSV must contain a 'text' column", 400

                # Clean text
                df['text'] = df['text'].apply(clean_text)

                # Predict if model exists
                if model:
                    try:
                        predictions = model.predict(df['text'])
                        df['label'] = predictions
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        # Fallback if prediction fails
                        if 'label' not in df.columns:
                            df['label'] = 1
                else:
                    print("Model not loaded")
                    if 'label' not in df.columns:
                        df['label'] = 1

                # Add src if missing
                if 'src' not in df.columns:
                    df['src'] = 'Upload'

                # Ensure ID
                if 'ID' not in df.columns:
                    df['ID'] = range(1, len(df) + 1)

                # Save to user specific file
                save_data(df)

                return redirect(url_for('index'))
            except Exception as e:
                print(f"Error processing file: {e}")
                return f"Error processing file: {e}", 500
    return render_template('upload.html')


@app.route('/validation')
@login_required
def validation():
    return render_template('validation.html')


@app.route('/api/validate', methods=['POST'])
@login_required
def validate_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        df = pd.read_csv(file)

        if 'text' not in df.columns or 'label' not in df.columns:
            return jsonify({'error': 'CSV файл должен содержать колонки "text" и "label"'}), 400

        # Clean text
        df['cleaned_text'] = df['text'].apply(clean_text)

        # Predict
        if model:
            predictions = model.predict(df['cleaned_text'])
        else:
            return jsonify({'error': 'Модель не загружена'}), 500

        # Calculate Metrics
        y_true = df['label'].astype(int)
        y_pred = predictions

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)

        return jsonify({
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'report': report
        })

    except Exception as e:
        print(f"Validation error: {e}")
        return jsonify({'error': f'Ошибка при обработке файла: {str(e)}'}), 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        users = load_users()

        # Mock auth
        if username in users and users[username]['password'] == password:
            user_data = users[username]
            session['user'] = {
                'username': username,
                'name': user_data['name'],
                'role': user_data['role']
            }
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Неверное имя пользователя или пароль')

    return render_template('login.html')


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    current_username = session['user']['username']
    users = load_users()
    user_data = users.get(current_username)

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profile':
            new_name = request.form.get('name')
            if new_name:
                user_data['name'] = new_name
                users[current_username]['name'] = new_name
                save_users(users)
                session['user']['name'] = new_name
                session.modified = True
                return render_template('settings.html', success='Профиль обновлен', user=user_data)

        elif action == 'change_password':
            current_pwd = request.form.get('current_password')
            new_pwd = request.form.get('new_password')
            confirm_pwd = request.form.get('confirm_password')

            if current_pwd != user_data['password']:
                return render_template('settings.html', error='Неверный текущий пароль', user=user_data)

            if new_pwd != confirm_pwd:
                return render_template('settings.html', error='Пароли не совпадают', user=user_data)

            user_data['password'] = new_pwd
            users[current_username]['password'] = new_pwd
            save_users(users)
            return render_template('settings.html', success='Пароль успешно изменен', user=user_data)

    return render_template('settings.html', user=user_data)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

