from flask import Flask, redirect, request, session, send_from_directory
from flask_cors import CORS
import requests
import lyricsgenius
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)
CORS(app)

# -------------------- CONFIGURATION --------------------
with open('config.json', 'r') as f:
    config = json.load(f)

genius_api_key = config["genius_api_key"]
spotify_client_id = config["spotify_client_id"]
spotify_client_secret = config["spotify_client_secret"]
genius = lyricsgenius.Genius(genius_api_key)

nltk.download('punkt')
nltk.download('stopwords')

vectorizer = TfidfVectorizer(stop_words='english')


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

@app.route('/login')
def login():
    """Redirect to Spotify login page."""
    auth_url = 'https://accounts.spotify.com/authorize'
    params = {
        'client_id': spotify_client_id,
        'response_type': 'code',
        'redirect_uri': 'http://localhost:5000/',
        'scope': 'playlist-read-private'
    }
    response = requests.get(auth_url, params=params)
    return redirect(response.url)


@app.route('/callback')
def callback():
    """Handle callback from Spotify."""
    code = request.args.get('code')
    token_url = 'https://accounts.spotify.com/api/token'
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': 'http://localhost:5000/',
        'client_id': spotify_client_id,
        'client_secret': spotify_client_secret
    }
    response = requests.post(token_url, data=data)
    token_info = response.json()
    session['access_token'] = token_info['access_token']
    return "Token Received!"  # Placeholder, can be changed to render a template or redirect.

@app.route('/playlists')
def playlists():
    access_token = session.get('access_token')
    headers = {'Authorization': f'Bearer {access_token}',}
    response = requests.get('https://api.spotify.com/v1/me/playlists', headers=headers)

    if response.status_code != 200:
        return {"error": "Failed to fetch playlists"}, 400

    playlists_data = response.json()
    enrich_playlists_with_track_details(playlists_data, headers)
    return playlists_data

@app.route('/sort')
def sort():
    method = request.args.get('method', default='bpm', type=str)
    playlist_data = playlists()

    if method == 'bpm':
        return playlist_data  # Already sorted by BPM in the 'playlists' endpoint
    elif method == 'camelot':
        return sort_tracks_camelot(playlist_data)
    elif method == 'wordplay':
        return sort_songs_by_similarity()
    else:
        return {"error": f"Unknown sorting method: {method}"}, 400

@app.route('/export', methods=['GET'])
def export():
    method = request.args.get('method')
    sorted_playlists = sort()

    filename = f"sorted_playlist_{method}.txt"
    with open(filename, 'w') as file:
        for playlist in sorted_playlists['items']:
            file.write(f"Playlist Name: {playlist['name']}\n")
            for item in playlist['tracks']['items']:
                track_info = item['track']
                file.write(f"Track Name: {track_info['name']}, BPM: {track_info['bpm']}, Camelot Key: {track_info['camelot']}\n")
            file.write("\n")
    
    return send_from_directory(directory='.', filename=filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)











##OLD CODE

# def enrich_playlists_with_track_details(playlists, headers):
#     """Add BPM, key and other track details to each song in the playlists."""
#     for playlist in playlists['items']:
#         for item in playlist['tracks']['items']:
#             enrich_track_details(item, headers)


# def enrich_track_details(item, headers):
#     """Add BPM, key, and other details to a single track."""
#     track_id = item['track']['id']
#     response = requests.get(f'https://api.spotify.com/v1/audio-features/{track_id}', headers=headers)
#     if response.status_code == 200:
#         audio_features = response.json()
#         enrich_track_with_audio_features(item, audio_features)


# def enrich_track_with_audio_features(item, audio_features):
#     """Incorporate audio features to the track's details."""
#     bpm = audio_features['tempo']
#     key = audio_features['key']
#     major_minor = audio_features['mode']
#     display_key = 'major' if major_minor == 1 else 'minor'
#     danceability = audio_features['danceability']
#     energy = audio_features['energy']
#     camelot = spotify_to_camelot(key, major_minor)
    
#     item['track'].update({
#         'bpm': bpm,
#         'key': key,
#         'major_minor': display_key,
#         'camelot': camelot,
#         'danceability': danceability,
#         'energy': energy,
#         'lyrics': fetch_and_process_lyrics(item['track']['name'], item['track']['artists'][0]['name'])
#     })

# # Other helper functions and routes like spotify_to_camelot, sort_tracks, etc. remain the same.

# def spotify_to_camelot(key, mode):
#     # Create a mapping from Spotify's key and mode to Camelot notation
#     camelot_notation = {
#         (0, 1): '8B', (1, 1): '3B', (2, 1): '10B', (3, 1): '5B',
#         (4, 1): '12B', (5, 1): '7B', (6, 1): '2B', (7, 1): '9B',
#         (8, 1): '4B', (9, 1): '11B', (10, 1): '6B', (11, 1): '1B',
#         (0, 0): '5A', (1, 0): '12A', (2, 0): '7A', (3, 0): '2A',
#         (4, 0): '9A', (5, 0): '4A', (6, 0): '11A', (7, 0): '6A',
#         (8, 0): '1A', (9, 0): '8A', (10, 0): '3A', (11, 0): '10A',
#     }
    
#     # If no key detected return None
#     if key == -1:
#         return None

#     return camelot_notation[(key, mode)]


# def sort():
#     method = request.args.get('method', default='bpm', type=str)
    
#     if method == 'bpm':
#         sorted_playlists = sort_tracks()
#     elif method == 'camelot':
#         sorted_playlists = sort_tracks_camelot()
#     elif method == 'wordplay':
#         sorted_playlists = sort_songs_by_lyrics()
#     else:
#         return {"error": f"Unknown sorting method: {method}"}, 400

#     return sorted_playlists


# # A function to fetch and process lyrics from Genius
# def fetch_and_process_lyrics(track_name, artist_name):
#     song = genius.search_song(track_name, artist_name)
#     if song is None or not song.lyrics or len(song.lyrics) < 50:  # Checking if lyrics are empty or too short
#         return None
#     lyrics = song.lyrics

#     # Process the lyrics
#     words = word_tokenize(lyrics.lower())
#     stop_words = set(stopwords.words('english'))
#     words = [word for word in words if word.isalnum() and word not in stop_words]
#     return words


# def update_lyrics_list_and_similarity_matrix():
#     global lyrics_list, song_names_list
#     lyrics_list = [" ".join(song['track']['lyrics']) for song in playlists()['items']]
#     song_names_list = [song['track']['name'] for song in playlists['items']]
#     tfidf_matrix = vectorizer.fit_transform(lyrics_list)
#     return cosine_similarity(tfidf_matrix)

# similarity_matrix = update_lyrics_list_and_similarity_matrix()

# def sort_songs_by_similarity(song_index):
#     song_similarities = similarity_matrix[song_index]
#     sorted_indexes = song_similarities.argsort()[::-1]
#     return sorted_indexes


# def sort_tracks_camelot():
#     playlists = playlists()
#     for playlist in playlists['items']:
#         playlist['tracks']['items'].sort(key=lambda x: (x['track']['camelot'][:-1], x['track']['camelot'][-1]))
#     return playlists


# @app.route('/export', methods=['GET'])
# def export():
#     method = request.args.get('method')
    
#     if method == 'bpm':
#         sorted_playlists = sort_tracks()
#     elif method == 'camelot':
#         sorted_playlists = sort_tracks_camelot()
#     elif method == 'wordplay':
#         sorted_playlists = sort_songs_by_lyrics()
#     else:
#         return {"error": f"Unknown sorting method: {method}"}, 400

#     filename = f"sorted_playlist_{method}.txt"
    
#     with open(filename, 'w') as file:
#         for playlist in sorted_playlists['items']:
#             file.write(f"Playlist Name: {playlist['name']}\n")
#             for item in playlist['tracks']['items']:
#                 track_info = item['track']
#                 file.write(f"Track Name: {track_info['name']}, BPM: {track_info['bpm']}, Camelot Key: {track_info['camelot']}\n")
#             file.write("\n")
    
#     return send_from_directory(directory='.', filename=filename, as_attachment=True)