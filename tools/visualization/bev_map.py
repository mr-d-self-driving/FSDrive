import numpy as np
import json
import pickle
import os
import os.path as osp
from PIL import Image

def vis_sd(osm_vectors ,image_dir):
        import os.path as osp
        scale_factor = 10
        car_img = Image.open('/home/zengshuang.zs/MapTRv2_vis/figs/lidar_car.png')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(2, 4))
        pc_range= [-15.0, -30.0,-10.0, 15.0, 30.0, 10.0]
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        for vector in osm_vectors:
            pts = vector
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])    
            plt.plot(x, y, color='green',linewidth=1,alpha=0.8,zorder=-1)
        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        plt.savefig(image_dir, bbox_inches='tight', format='png',dpi=1200)
        plt.close()  

def load_pred_from_json(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def convert(input_string):
    # 初始化最终的嵌套列表
    result = []

    # 按分号分割字符串，得到每个组
    groups = input_string.split(';')

    # 遍历每个组
    for group in groups:
        group = group.strip().strip('()')  # 去除空格和括号
        # 按"and"分割，得到每个坐标对
        pairs = group.split('and')
        # 初始化当前组的坐标列表
        coordinates = []
        for pair in pairs:
            pair = pair.strip()  # 去除前后空格
            x, y = pair.split(',')  # 按逗号分割x和y
            # 将字符串转换为浮点数并添加到当前坐标列表
            coordinates.append([float(x), float(y)])
        # 将当前组的坐标列表添加到最终结果中
        result.append(coordinates)
    return result

pred = load_pred_from_json("/home/zengshuang.zs/output/llm/v2.12/predict/generated_predictions.jsonl")
token_traj =  json.load(open('/home/zengshuang.zs/LLaMA-Factory/data/val_motion.json', 'r'))
path=f"/home/zengshuang.zs/LLaMA-Factory/plan_eval/vis_map2.12"
os.makedirs(path, exist_ok=True)
num=0
# nuscenes_map=pickle.load(open('/home/zengshuang.zs/LLaMA-Factory/create_data/nuscenes_map.pkl', 'rb'))

for i, traj in enumerate(token_traj):
    try:
        out_path=osp.join(path, f"{traj['id']}.jpg")
        result = convert(pred[i]['predict'])
        vis_sd(result, out_path)
    except:
        num+=1
print(num)
