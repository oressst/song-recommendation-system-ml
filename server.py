from flask import Flask, jsonify, request
from scipy.spatial import distance
from sklearn import preprocessing
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
# Load and prepare data once when starting the server
def initialize_data():
    global df, data_norm
    df = pd.read_csv('data/dataset.csv')
    df = df.drop(['time_signature', 'key'], axis=1)
    df.drop_duplicates(subset=['track_id'], inplace=True)

    # Normalize the data
    scaler = preprocessing.MinMaxScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    data_norm = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), 
                            columns=numerical_cols, 
                            index=df['track_id'])

initialize_data()

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    track_name = request.args.get('song')
    if not track_name:
        return jsonify({"error": "Song name is required"}), 400
        
    try:
        # Get target track information
        track_id = df[df['track_name'] == track_name][['track_id']]
        if track_id.empty:
            return jsonify({"error": "Song not found"}), 404
            
        track_id = track_id.values[0][0]
        target_track = list(data_norm.loc[track_id])
        
        # Calculate Euclidean distances
        data_result = pd.DataFrame()
        data_result['euclidean'] = [distance.euclidean(list(obj), target_track) 
                                   for _, obj in data_norm.iterrows()]
        data_result['track_id'] = data_norm.index
        
        # Get top 5 recommendations
        data_rec = data_result.sort_values(by=['euclidean']).iloc[:6]
        
        # Get track details
        data_init = df.set_index('track_id')
        track_list = []
        for i in data_rec['track_id']:
            if i in df['track_id'].values:
                track_info = data_init.loc[i, ['track_name', 'artists']]
                track_list.append({
                    "track_name": track_info['track_name'],
                    "artist": track_info['artists']
                })
        
        return jsonify({
            "currently_playing": track_list[0],
            "top_recommendation": track_list[1],
            "other_recommendations": track_list[2:]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)