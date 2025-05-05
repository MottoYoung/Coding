import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance
class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70  # 最大允许的距离
    def assign_ball_to_player(self,players_tracks,ball_bbox):
        ball_position=get_center_of_bbox(ball_bbox)
        minumum_distance=sys.maxsize
        assigned_player=-1
        for track_id,track in players_tracks.items():
            player_bbox=track['bbox']

            distance_left=measure_distance((player_bbox[0],player_bbox[-1]),ball_position) #球的位置是x_center,y2,所以球员左脚是x1,y2
            distance_right=measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance=min(distance_left,distance_right)

            if distance<self.max_player_ball_distance and distance<minumum_distance:
                minumum_distance=distance
                assigned_player=track_id
        return assigned_player