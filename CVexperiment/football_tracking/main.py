from utils import read_video, write_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camear_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimate import SpeedAndDistance_Estimate
def main():
    #read video
    video_path = "input_videos/B1606b0e6_1 (30).mp4"
    video_frames = read_video(video_path)
    video_name = video_path.split("/")[-1].split(".")[0]


    # inintialize tracker
    #目前看来v11_100最好
    model_path='models/player_detection/best_v11_100.pt'
    tracker=Tracker(model_path=model_path)
    model_name=model_path.split("/")[-1].split(".")[0]

    tracks=tracker.get_object_tracks(video_frames,
                                     read_from_stub=True,
                                     stub_path=f'stubs/track_stubs_{video_name}_{model_name}.pkl')
    
    #interpolate ball positions   插值必须在计算球的位置之前进行，否则位置为空距离计算会报错
    tracks["ball"]=tracker.interplate_ball_positions(tracks["ball"])

    #get object position   这个是为了计算速度算的bbox的下边界的中点的位置
    tracker.add_position_to_tracks(tracks)


    ###camera movement estimation
    camera_movement_estimator=CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame=camera_movement_estimator.get_camera_movement(video_frames,
                                                              read_from_stub=True,
                                                              stub_path=f'stubs/camera_movement_{video_name}_{model_name}.pkl')
    camera_movement_estimator.add_adjust_position_to_tracks(tracks,camera_movement_per_frame)
   
    # add_adjust_position_to_tracks=tracker.add_adjust_position_to_tracks(tracks,
    #view transformer
    view_transformer=ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    #speed and distance estimation
    speed_and_distance_estimator=SpeedAndDistance_Estimate()
    speed_and_distance_estimator.add_seppd_and_distance_to_tracks(tracks)




    #assign player teams
    team_assigner=TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num,player_tracks in enumerate(tracks['players']):
        for track_id,track in player_tracks.items():
            team=team_assigner.get_player_team(video_frames[frame_num],
                                               track['bbox'],
                                               track_id)
            tracks['players'][frame_num][track_id]['team']=team
            tracks['players'][frame_num][track_id]['color']=team_assigner.team_colors[team]

    #assign ball to player
    player_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num,player_track in enumerate(tracks['players']):
        ball_bbox=tracks["ball"][frame_num]['bbox']
        assigned_player=player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player!=-1:
            tracks['players'][frame_num][assigned_player]['has_ball']=True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if len(team_ball_control)>0:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control=np.array(team_ball_control)
    
    # #save cropped image of a player
    # for track_id,player in tracks['players'][0].items():
    #     bbox=player['bbox']
    #     frame=video_frames[0]

    #     #crop image of player
    #     cropped_image=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
    #     #save image
    #     cv2.imwrite(f"output_videos/crooped_image.jpg",cropped_image)
    #     break

    # DRAW player and ball positions
    output_video_frames=tracker.draw_annotations(video_frames,tracks,team_ball_control)

    # draw camera movement
    output_video_frames=camera_movement_estimator.draw_camera_movement(output_video_frames,
                                                                       camera_movement_per_frame)
    # draw speed and distance
    output_video_frames=speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,
                                                                    tracks)
    # save video
    output_video_path = f"output_videos/{video_name}_{model_name}_output.avi"
    write_video(output_video_frames, output_video_path)
if __name__=="__main__":
    main()
