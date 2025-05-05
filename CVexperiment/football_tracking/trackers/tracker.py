from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width,get_foot_position

class Tracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()
        self.stub_path=None
    def add_position_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                if object=="players" or object=="referees":
                    for track_id,track_info in track.items():
                            bbox=track_info['bbox']
                            position=get_foot_position(bbox)
                            tracks[object][frame_num][track_id]['position']=position
                if object=="ball" and track.get("bbox")!=None:
                    bbox=track.get('bbox')
                    position=get_center_of_bbox(bbox)
                    tracks[object][frame_num]['position']=position

    def interplate_ball_positions(self,ball_positions):
        ball_positions=[x.get('bbox',[]) for x in ball_positions]
        df_ball_positions=pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions=df_ball_positions.interpolate() #可以插大部分的值
        df_ball_positions=df_ball_positions.bfill()  # Backward fill （如果前面几帧没有值，回填一下

        ball_positions=[{"bbox":x} for x in df_ball_positions.to_numpy().tolist()]

        #  # 只有当stub_path存在时才尝试保存
        # if self.stub_path is not None:
        #     try:
        #         # 先读取原始数据
        #         with open(self.stub_path, 'rb') as f:
        #             all_tracks = pickle.load(f)
                
        #         # 更新球的位置数据
        #         all_tracks["ball"] = ball_positions
                
        #         # 保存回文件
        #         with open(self.stub_path, 'wb') as f:
        #             pickle.dump(all_tracks, f)
        #         print(f"插值后的球位置数据已保存至: {self.stub_path}")
        #     except Exception as e:
        #         print(f"保存插值数据失败: {e}")

        return ball_positions

    def detect_frames(self,frames):
        batch_size=20 # 20 frames at a time
        detections=[]
        for i in range(0,len(frames),batch_size):
            detections_batch=self.model.predict(frames[i:i+batch_size],conf=0.1) #高于此置信度的才会被检测到
            #用predict而非track是因为守门员数据较少，不稳定，在守门员和球员之间跳来跳去,用一个supervision来追踪,然后将其转换为球员

            detections+=detections_batch
            # break
        return detections
    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):
        self.stub_path=stub_path

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks=pickle.load(f)
            return tracks
            

        detections=self.detect_frames(frames)

        tracks={
            "players":[], # player tracks
            "referees":[], #referee tracks
            "ball":[] # ball tracks
        }
        # for each frame in yolo detections
        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_idx={v:k for k,v in cls_names.items()}


            #convert to supervision format
            detection_supervision=sv.Detections.from_ultralytics(detection)

            #conver goalkeeper to player

            for object_idx,class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id]=="goalkeeper":
                    detection_supervision.class_id[object_idx]=cls_names_idx["player"]
        
            #tracker objects(record every player's id and position in the frame,detection has note)
            #detection_surpervision is the detection with goalkeeper converted to player
            detection_with_tracks=self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            #record every object's id and position in the frame
            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]
            
                if cls_id==cls_names_idx["player"]:
                    tracks["players"][frame_num][track_id]={"bbox":bbox}
                if cls_id==cls_names_idx["referee"]:
                    tracks["referees"][frame_num][track_id]={"bbox":bbox}
                # if cls_id==cls_names_idx["ball"]:
                #     tracks["ball"][frame_num]={"bbox":bbox}
            for frame_detection in detection_supervision:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                if cls_id==cls_names_idx["ball"]:
                    # print(f"frame_num:{frame_num}",frame_detection)
                    # break
                    tracks["ball"][frame_num]={"bbox":bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        return tracks
    def draw_ellipse(self,frame,bbox,color,track_id=None,thickness=2):
        y2=int(bbox[3])

        x_center,_=get_center_of_bbox(bbox)
        width=get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center=(x_center,y2),
                    axes=(int(width/2),int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_4)
        ## 画球员编号
        rectangle_width=40
        rectangle_height=20
        x1_rect=x_center-rectangle_width//2
        x2_rect=x_center+rectangle_width//2
        y1_rect=(y2-rectangle_height//2) + 15
        y2_rect=(y2+rectangle_height//2) + 15
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect)),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            x1_text=x1_rect+12
            if track_id > 99:
                x1_text-=10
            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text),int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        color=(0,0,0),
                        thickness=2,
                        fontScale=0.6
                        ) 
        return frame
    def draw_traingle(self,frame,bbox,color):
        y=int(bbox[1])
        x,_=get_center_of_bbox(bbox)

        triangle_points=np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)
        return frame
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # semi_transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha=0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        #draw team ball control text
        team_ball_control_till_frame=team_ball_control[:frame_num+1]

        team_1_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        team_1_percentage=team_1_num_frames/(team_1_num_frames+team_2_num_frames)*100
        team_2_percentage=team_2_num_frames/(team_1_num_frames+team_2_num_frames)*100

        cv2.putText(frame,f"Team 1 Ball Control: {team_1_percentage:.2f}%",(1400,900),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),thickness=3)
        cv2.putText(frame,f"Team 2 Ball Control: {team_2_percentage:.2f}%",(1400,950),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),thickness=3)

        return frame


    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_frames=[]
        # print("length of frame",len(video_frames))
        # print("length of tracks",len(tracks['players']))
        # print("shape of input video frames",video_frames[0].shape)
        for frame_num,frame in enumerate(video_frames):
            frame=frame.copy()

            player_dict=tracks['players'][frame_num]
            referees_dict=tracks['referees'][frame_num]
            ball_dict=tracks['ball'][frame_num]
            # Draw players
            for track_id,player in player_dict.items():
                color=player.get('color',(0,0,255))
                frame=self.draw_ellipse(frame,player['bbox'],color,track_id,thickness=2)
                if player.get('has_ball',False):
                    frame=self.draw_traingle(frame,player['bbox'],(0,0,255))
             # Draw rteferees
            for track_id,referee in referees_dict.items():
                frame=self.draw_ellipse(frame,referee['bbox'],(0,255,255),track_id,thickness=2)
            # Draw ball
            if 'bbox' in ball_dict and len(ball_dict['bbox']) == 4:
                frame=self.draw_traingle(frame, ball_dict['bbox'], (0,255,0))

            # Draw team ball control
            frame=self.draw_team_ball_control(frame,frame_num,team_ball_control)
            output_frames.append(frame)
        return output_frames
