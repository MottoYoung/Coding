import numpy as np
import cv2
class ViewTransformer():
    def __init__(self):
        court_width=68
        # court_length=42  #只取了中间的八个草格画矩形
        court_length=31.5  #只取了中间的八个草格画矩形

        # 距离根据视频调整，梯形对应长方形
        self.pixel_verticies=np.array([
            #test32
            # [699,255],
            # [1309,255],
            # [223,815],     
            # [1902,915]
            #input_videos\B1606b0e6_1 (30).mp4
            [478,181],
            [1188,163],
            [64,874],
            [1586,833]
        ])
        self.target_verticies=np.array(
            [   [0,0],    ##左上角
                [court_length,0],  ##右上角
                [0,court_width],   ##左下角
                [court_length,court_width]  ##右下角
            ]
        )
        self.pixel_verticies=self.pixel_verticies.astype(np.float32)
        self.target_verticies=self.target_verticies.astype(np.float32)

        self.perspective_transform_matrix=cv2.getPerspectiveTransform(self.pixel_verticies,self.target_verticies)
    def transfrom_point(self,point):
        p=(int(point[0]),int(point[1]))
        is_inside=cv2.pointPolygonTest(self.pixel_verticies,p,False)>=0
        if not is_inside:
            return None
        
        reshaped_point=point.reshape(-1,1,2).astype(np.float32)
        transformed_point=cv2.perspectiveTransform(reshaped_point,self.perspective_transform_matrix)

        return transformed_point.reshape(-1,2)
            

    def add_transformed_position_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_num,track in enumerate(object_tracks):
                if object=="players" or object=="referees":
                    for track_id,track_info in track.items():
                            position=track_info['position_adjusted']
                            position=np.array(position)
                            position_transformed=self.transfrom_point(position)
                            if position_transformed is None:
                            #    print(f"position_transformed is None, object: {object}, track_id: {track_id}, frame_num: {frame_num}")
                               tracks[object][frame_num][track_id]['position_transformed'] = position.tolist()  # 使用原始位置作为默认值
                            else:
                                position_transformed=position_transformed.squeeze().tolist()
                                tracks[object][frame_num][track_id]['position_transformed']=position_transformed
                if object=="ball" and track.get("position_adjusted")!=None:
                    position=track.get('position_adjusted',[])
                    position=np.array(position)
                    position_transformed=self.transfrom_point(position)
                    if position_transformed is None:
                        # 同样为球设置默认值
                        tracks[object][frame_num]['position_transformed'] = position.tolist()
                    else:
                        position_transformed=position_transformed.squeeze().tolist()
                        tracks[object][frame_num]['position_transformed']=position_transformed