from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "http://localhost:3000",  # React app URL
        "methods": ["GET"],
        "allow_headers": ["Content-Type"]
    }
})

# Load and prepare data
def initialize_data():
    global df, normalized_data, track_mapping
    df = pd.read_csv('data/dataset.csv')
    df = df.drop(['time_signature', 'key'], axis=1)
    df.drop_duplicates(subset=['track_id'], inplace=True)  # Remove duplicate tracks

    # Normalize numerical data
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    normalized_data = pd.DataFrame(
        scaler.fit_transform(df[numerical_cols]),
        columns=numerical_cols,
        index=df['track_id']
    )

    # Map track_id to track_name and artist
    track_mapping = df.set_index('track_id')[['track_name', 'artists', 'album_name']]

initialize_data()

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    track_name = request.args.get('song')
    if not track_name:
        return jsonify({"error": "Song name is required"}), 400

    try:
        # Get the track_id for the input song
        track_id_row = df[df['track_name'] == track_name][['track_id']]
        if track_id_row.empty:
            return jsonify({"error": "Song not found"}), 404
        
        track_id = track_id_row.values[0][0]

        # Extract feature vector for the input song
        track_features = normalized_data.loc[track_id]

        # Calculate cosine similarity for all tracks
        similarities = normalized_data.apply(lambda x: 1 - cosine(track_features, x), axis=1)
        most_similar = similarities.sort_values(ascending=False).head(6)  # Top 5 + input song

        # Prepare recommendation list
        recommendations = []
        for rec_id in most_similar.index:
            track_info = track_mapping.loc[rec_id]
            recommendations.append({
                "track_name": track_info['track_name'],
                "artist": track_info['artists'],
                "album_name": track_info['album_name']
                
            })

        # Respond with recommendations
        return jsonify({
            "currently_playing": recommendations[0],  # First is the input song
            "recommendations": recommendations[1:]  # The rest are suggestions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize_data()
    app.run(debug=True)
