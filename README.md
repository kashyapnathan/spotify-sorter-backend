### How the Backend Works

The backend of the Spotify Sorter application is built using Flask and integrates with the Spotify and Genius APIs. It provides endpoints for fetching and sorting Spotify playlists. Key functionalities include:

- **Fetching Playlist Data**: Retrieves playlist details from Spotify and enriches tracks with audio features and lyrics.
- **Sorting Playlists**: Sorts tracks based on BPM, Camelot notation, or lyric similarity.
- **Exporting Data**: Allows exporting of sorted playlists to a text file.
