from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import os
import re

base_dir = "C:/Coding/CVexperiment/football_tracking/runs"
# 确保基本目录存在
os.makedirs(base_dir, exist_ok=True)

# 查找所有predict开头的目录并确定最大索引值
max_index = 0
predict_pattern = re.compile(r'^predict(\d*)$')
if os.path.exists(base_dir):
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            match = predict_pattern.match(item)
            if match:
                # 如果匹配到 predict 或 predictX 格式
                num_str = match.group(1)
                index = int(num_str) if num_str else 0  # 如果没有数字，则为predict默认为0
                max_index = max(max_index, index)

# 创建新的保存目录
save_dir = os.path.join(base_dir, f"predict{max_index + 1}")
print(f"将结果保存至: {save_dir}")

DEFAULT_CFG.save_dir = save_dir
model = YOLO('models/player_detection/best_100.pt')
results = model.predict("input_videos/B1606b0e6_1 (30).mp4", save=True)
print(results[0])
print("result saved in", results[0].save_dir)
print("==============================")
for box in results[0].boxes:
    print(box)