import sys
sys.path.append('../')

from f1tenth_gym.f110_env import F110Env
from src.envs.wrapper import F110Wrapper
from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT

def calculate_index_based_color(scan, num_sections=5):
    """
    LiDARのインデックスに基づいて色分けを行う
    - num_sections: 分割するセクションの数
    """
    num_points = len(scan)
    section_size = num_points // num_sections
    
    colors = np.zeros((num_points, 4))
    color_map = plt.cm.jet(np.linspace(0, 1, num_sections))

    for i in range(num_sections):
        start = i * section_size
        end = (i + 1) * section_size if i < num_sections - 1 else num_points
        colors[start:end] = color_map[i]
    
    return colors

map_name = 'Austin'
map_ext = '.png'
speed = 8.0
downsample = 1
use_dynamic_speed = True
a_lat_max = 3
smooth_sigma = 2

map_manager = MapManager(
    map_name=map_name,
    map_ext=map_ext,
    speed=speed,
    downsample=downsample,
    use_dynamic_speed=use_dynamic_speed,
    a_lat_max=a_lat_max,
    smooth_sigma=smooth_sigma
)

vehicle_param = {
    'mu': 1.0489,
    'C_Sf': 4.718,
    'C_Sr': 5.4562,
    'lf': 0.15875,
    'lr': 0.17145,
    'h': 0.074,
    'm': 3.74,
    'I': 0.04712,
    's_min': -0.4,
    's_max': 0.4,
    'sv_min': -3.2,
    'sv_max': 3.2,
    'v_switch': 7.319,
    'a_max': 9.51,
    'v_min': -5.0,
    'v_max': 10.0,
    'width': 0.31,
    'length': 0.58
}

num_beams = 1080
num_agents = 1
## 公式のベース環境
env = F110Env(map=map_manager.map_path, map_ext=map_ext, num_beams=num_beams, num_agents=num_agents, params=vehicle_param)
## 自作のラッパー
env = F110Wrapper(env, map_manager=map_manager)

from src.planner.purePursuit import PurePursuitPlanner
wheelbase = 0.33
lookahead = 0.6
gain = 0.2
max_reacquire = 20.0

planner = PurePursuitPlanner(
    wheelbase=wheelbase,
    map_manager=map_manager,
    lookahead=lookahead,
    gain=gain,
    max_reacquire=max_reacquire
)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- インタラクティブな極座標プロットのセットアップ ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'polar': True})
ax.set_theta_direction(1)  
ax.set_theta_offset(-np.pi / 2)  
ax.set_ylim(0, 30.0)  # 最大30mまで描画

# 角度の設定 (-135°から135°まで1080分割)
num_points = 1080
angles = np.linspace(-135, 135, num_points) * (np.pi / 180.0)

# --- モードの設定（'gradient' または 'index_based' を手動で選ぶ） ---
mode = 'gradient'  # 'gradient' に切り替えると通常のグラデーション表示

# カラーマップの設定
gradient_colors = plt.cm.jet(np.linspace(0, 1, num_points))

# 初期プロットの設定
if mode == 'gradient':
    scat = ax.scatter(angles, np.zeros(num_points), c=gradient_colors, s=2)
else:
    scat = ax.scatter(angles, np.zeros(num_points), c='gray', s=2)

plt.show()

# メインループ
max_steps = 3000
for i in range(len(MAP_DICT)):
    map = MAP_DICT[i]
    env.update_map(map_name=map, map_ext=map_ext)
    obs, info = env.reset()
    done = False

    for step in range(max_steps):
        # --- LiDARデータの取得と更新 ---
        scan = obs['scans'][0]

        # --- モードに応じたカラー設定 ---
        if mode == 'gradient':
            scat.set_color(gradient_colors)
        elif mode == 'index_based':
            index_colors = calculate_index_based_color(scan)
            scat.set_color(index_colors)

        # 極座標プロットの更新
        scat.set_offsets(np.c_[angles, scan])
        plt.pause(0.001)

        # --- アクションの計算 ---
        actions = []
        for agent_id in range(num_agents):
            steer, speed = planner.plan(obs, id=agent_id)
            steer, speed = 0.0, 0.0
            action = [steer, speed]
            actions.append(action)

        # --- 環境ステップの実行 ---
        next_obs, reward, terminated, truncated, info = env.step(np.array(actions))

        # --- 環境の描画 ---
        env.render(mode='human')
        
        # --- エピソード終了判定 ---
        if terminated or truncated:
            break

        # --- 観測の更新 ---
        obs = next_obs

