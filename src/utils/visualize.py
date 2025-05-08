import os
import yaml
import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt

def visualize_speed_map(map_manager, figsize=(10,10), save_path=None):
    """YAML指定＋origin/resolution考慮でマップ上に速度可視化"""
    # YAML からマップ設定読み込み
    with open(map_manager.map_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    image_file = cfg['image']
    resolution = cfg['resolution']  # [m/pixel]
    origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]

    # 画像読み込み
    img_path = os.path.join(map_manager.map_base_dir, image_file)
    img = plt.imread(img_path)
    height, width = img.shape[:2]

    # ワールド座標→ピクセル座標変換（Y軸反転補正）
    wpts = map_manager.waypoints[:, :2]
    px = (wpts[:, 0] - origin_x) / resolution
    py = height - (wpts[:, 1] - origin_y) / resolution

    # 描画
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper')
    sc = ax.scatter(px, py, c=map_manager.waypoints[:, 2], cmap='bwr', s=10)

    # カラーバー
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Speed [m/s]')
    ax.set_title(f"Speed Visualization: {map_manager.map_name}")
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def visualize_curve_class_map(map_manager, figsize=(10,10), save_path=None):
    """マップ上にカーブのクラス分類結果を可視化（速度ではなくクラスで色分け）"""
    # YAML からマップ設定読み込み
    with open(map_manager.map_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    image_file = cfg['image']
    resolution = cfg['resolution']  # [m/pixel]
    origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]

    # 画像読み込み
    img_path = os.path.join(map_manager.map_base_dir, image_file)
    img = plt.imread(img_path)
    height, width = img.shape[:2]

    # ワールド座標→ピクセル座標変換（Y軸反転）
    wpts = map_manager.waypoints[:, :2]
    px = (wpts[:, 0] - origin_x) / resolution
    py = height - (wpts[:, 1] - origin_y) / resolution

    # クラス情報
    if not hasattr(map_manager, 'curve_classes'):
        raise AttributeError("MapManagerに 'curve_classes' が存在しません。_compute_speeds で分類が行われているか確認してください。")
    classes = map_manager.curve_classes

    # 描画
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper')
    sc = ax.scatter(px, py, c=classes, cmap='tab10', s=10)

    # カラーバー
    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(classes.min(), classes.max()+1))
    cbar.set_label('Curve Class (0=Straight, ↑Tighter)')
    ax.set_title(f"Curve Class Visualization: {map_manager.map_name}")
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def visualize_trajectory(map_name, base_dir='./f1tenth_gym/maps', figsize=(10, 10), save_path=None):
    """
    指定されたマップの走行軌跡と速度をPNGマップ上に可視化
    
    Args:
        map_name (str): マップ名（例: 'Austin'）
        base_dir (str): マップファイルが存在するベースディレクトリのパス
        figsize (tuple): プロットのサイズ
        save_path (str): 保存先のパス（指定しない場合は表示のみ）
    """
    # パスの構築
    yaml_path = os.path.join(base_dir, map_name, f"{map_name}_map.yaml")
    csv_path = os.path.join(base_dir, map_name, f"{map_name}_trajectory.csv")
    
    # --- YAML の読み込み ---
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    image_file = cfg['image']
    resolution = cfg['resolution']
    origin_x, origin_y = cfg['origin'][0], cfg['origin'][1]
    
    # 画像の読み込み
    img_path = os.path.join(base_dir, map_name, image_file)
    img = plt.imread(img_path)
    height, width = img.shape[:2]

    # --- CSV の読み込み ---
    x_coords, y_coords, velocities = [], [], []

    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_coords.append(float(row['x']))
            y_coords.append(float(row['y']))
            velocities.append(float(row['velocity']))

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    velocities = np.array(velocities)

    # ワールド座標 → ピクセル座標変換（Y軸反転補正）
    px = (x_coords - origin_x) / resolution
    py = height - (y_coords - origin_y) / resolution

    # --- 描画 ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, origin='upper')
    sc = ax.scatter(px, py, c=velocities, cmap='bwr', s=5)

    # カラーバー
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Speed [m/s]')
    ax.set_title(f"Trajectory Visualization: {map_name}")
    ax.axis('off')

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()