import pickle
import os

path="stubs/track_stubs.pkl"
if os.path.exists(path):
    with open(path,'rb') as f:
        tracks=pickle.load(f)
        with open("stubs/track_stubs.txt",'w') as f:
            for frame_num,track in enumerate(tracks["players"]):
                f.write(f"frame_num: {frame_num}\n")
                for track_id,player in track.items():
                    bbox=player['bbox']
                    f.write(f"track_id: {track_id}, bbox: {bbox}\n")
                f.write("\n")
            f.write("\n")
            for frame_num,track in enumerate(tracks["referees"]):
                f.write(f"frame_num: {frame_num}\n")
                for track_id,player in track.items():
                    bbox=player['bbox']
                    f.write(f"track_id: {track_id}, bbox: {bbox}\n")
                f.write("\n")
            f.write("\n")
            for frame_num,track in enumerate(tracks["ball"]):
                f.write(f"frame_num: {frame_num}\n")
                for track_id,player in track.items():
                    bbox=player['bbox']
                    f.write(f"track_id: {track_id}, bbox: {bbox}\n")
                f.write("\n")
            f.write("\n")
        
    print("tracks loaded")
else:
    print("tracks not found")
    