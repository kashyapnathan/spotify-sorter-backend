from flask import Flask, jsonify, request, session, send_from_directory
from flask_cors import CORS
import requests
import lyricsgenius
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging

app = Flask(__name__)
CORS(app)

# -------------------- CONFIGURATION --------------------
with open('config.json', 'r') as f:
    config = json.load(f)

genius_api_key = config["genius_api_key"]
spotify_client_id = config["spotify_client_id"]
app_secret_key = config["app.secret_key"]
spotify_client_secret = config["spotify_client_secret"]
genius = lyricsgenius.Genius(genius_api_key)

nltk.download('punkt')
nltk.download('stopwords')

vectorizer = TfidfVectorizer(stop_words='english')
app.secret_key = app_secret_key

# -------------------- HELPER FUNCTIONS --------------------
def spotify_to_camelot(key, mode):
    camelot_notation = {
        (0, 1): '8B', (1, 1): '3B', (2, 1): '10B', (3, 1): '5B',
        (4, 1): '12B', (5, 1): '7B', (6, 1): '2B', (7, 1): '9B',
        (8, 1): '4B', (9, 1): '11B', (10, 1): '6B', (11, 1): '1B',
        (0, 0): '5A', (1, 0): '12A', (2, 0): '7A', (3, 0): '2A',
        (4, 0): '9A', (5, 0): '4A', (6, 0): '11A', (7, 0): '6A',
        (8, 0): '1A', (9, 0): '8A', (10, 0): '3A', (11, 0): '10A',
    }  # Existing mapping
    return camelot_notation.get((key, mode), None)

def fetch_and_process_lyrics(track_name, artist_name):
    song = genius.search_song(track_name, artist_name)
    if not song or not song.lyrics or len(song.lyrics) < 50:
        return None
    return process_lyrics(song.lyrics)

def process_lyrics(lyrics):
    words = word_tokenize(lyrics.lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word.isalnum() and word not in stop_words]

def enrich_track_with_audio_features(item, audio_features):
    item['track'].update({
        'bpm': audio_features['tempo'],
        'key': audio_features['key'],
        'major_minor': 'major' if audio_features['mode'] == 1 else 'minor',
        'camelot': spotify_to_camelot(audio_features['key'], audio_features['mode']),
        'danceability': audio_features['danceability'],
        'energy': audio_features['energy'],
        'lyrics': fetch_and_process_lyrics(item['track']['name'], item['track']['artists'][0]['name'])
    })

def enrich_track_details(item, headers):
    track_id = item['track']['id']
    response = requests.get(f'https://api.spotify.com/v1/audio-features/{track_id}', headers=headers)
    if response.status_code == 200:
        audio_features = response.json()
        enrich_track_with_audio_features(item, audio_features)

def enrich_playlists_with_track_details(playlists, headers):
    for playlist in playlists['items']:
        for item in playlist['tracks']['items']:
            enrich_track_details(item, headers)

def sort_tracks_by_bpm(playlists):
    for playlist in playlists['items']:
        playlist['tracks']['items'].sort(key=lambda x: x['track']['bpm'])
    return playlists

def sort_songs_by_similarity():
    lyrics_list = [" ".join(song['track']['lyrics']) for playlist in playlists()['items'] for song in playlist['tracks']['items']]
    tfidf_matrix = vectorizer.fit_transform(lyrics_list)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return sorted(range(len(lyrics_list)), key=lambda x: -similarity_matrix[x])

def sort_tracks_camelot(playlists):
    for playlist in playlists['items']:
        playlist['tracks']['items'].sort(key=lambda x: (x['track']['camelot'][:-1], x['track']['camelot'][-1]))
    return playlists

# -------------------- ROUTES --------------------

@app.route('/get_app_token')
def get_app_token():
    token_url = 'https://accounts.spotify.com/api/token'
    data = {
        'grant_type': 'client_credentials',
        'client_id': spotify_client_id,
        'client_secret': spotify_client_secret
    }
    response = requests.post(token_url, data=data)
    token_info = response.json()
    return jsonify({"access_token": token_info['access_token']})


@app.route('/fetch_playlist/<playlist_id>')
def fetch_playlist(playlist_id):
    access_token = get_app_token().json['access_token']  
    headers = {"Authorization": f"Bearer {access_token}"}
    playlist_url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
    response = requests.get(playlist_url, headers=headers)

    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch playlist"}), 400

    playlist_data = response.json()
    enrich_playlists_with_track_details(playlist_data, headers)
    return jsonify(playlist_data)


@app.route('/playlists')
def playlists():
    access_token = session.get('access_token')
    headers = {'Authorization': f'Bearer {access_token}',}
    response = requests.get('https://api.spotify.com/v1/me/playlists', headers=headers)

    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch playlists"}), 400

    playlists_data = response.json()
    enrich_playlists_with_track_details(playlists_data, headers)
    return jsonify(playlists_data)

@app.route('/sort/<playlist_id>')
def sort(playlist_id):
    method = request.args.get('method', default='bpm', type=str)
    playlist_data = fetch_playlist(playlist_id)

    if method == 'bpm':
        return sort_tracks_by_bpm(playlist_data)
    elif method == 'camelot':
        return sort_tracks_camelot(playlist_data)
    elif method == 'wordplay':
        return sort_songs_by_similarity()
    else:
        return {"error": f"Unknown sorting method: {method}"}, 400

@app.route('/export/<playlist_id>', methods=['GET'])
def export(playlist_id):
    method = request.args.get('method')
    sorted_playlists = sort(playlist_id)

    filename = f"sorted_playlist_{method}.txt"
    with open(filename, 'w') as file:
        file.write(f"Playlist Name: {sorted_playlists['name']}\n")
        for item in sorted_playlists['tracks']['items']:
            track_info = item['track']
            file.write(f"Track Name: {track_info['name']}, BPM: {track_info['bpm']}, Camelot Key: {track_info['camelot']}\n")
        file.write("\n")
    
    return send_from_directory(directory='.', filename=filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

