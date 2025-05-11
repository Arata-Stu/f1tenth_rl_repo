import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from f1tenth_gym.maps.map_manager import MapManager, MAP_DICT
from src.envs.envs import make_env
from src.planner.purePursuit import PurePursuitPlanner
from src.utils.helper import ScanBuffer, convert_scan, convert_action
from src.agents.agent import get_agent

@hydra.main(config_path="config", config_name="eval", version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(cfg))
    print('---------------------------')

    # --- 環境／プランナ／エージェント等の初期化 ---
    map_manager = MapManager(
        map_name=cfg.envs.map.name,
        map_ext=cfg.envs.map.ext,
        speed=cfg.envs.map.speed,
        downsample=cfg.envs.map.downsample,
        use_dynamic_speed=cfg.envs.map.use_dynamic_speed,
        a_lat_max=cfg.envs.map.a_lat_max,
        smooth_sigma=cfg.envs.map.smooth_sigma
    )
    env = make_env(cfg.envs, map_manager, cfg.vehicle)
    planner = PurePursuitPlanner(
        wheelbase=cfg.planner.wheelbase,
        map_manager=map_manager,
        lookahead=cfg.planner.lookahead,
        gain=cfg.planner.gain,
        max_reacquire=cfg.planner.max_reacquire,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- モデルの読み込み ---
    agent = get_agent(agent_cfg=cfg.agent, device=device)
    model_path = os.path.join(cfg.ckpt)
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    scan_buffer = ScanBuffer(
        frame_size=cfg.envs.num_beams,
        num_scan=cfg.scan_n,
        target_size=cfg.dowansample_beam
    )

    # --- 評価ループ ---
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    # 初期スキャンの登録
    scan = convert_scan(obs['scans'][0], cfg.envs.max_beam_range)
    scan_buffer.add_scan(scan)

    for step in range(cfg.num_steps):
        # 行動選択（エージェントとプランナ）
        actions = []
        for i in range(cfg.envs.num_agents):
            if i == 0:
                state = scan_buffer.get_concatenated_tensor()
                nn_action_dict = agent.select_action(state, evaluate=True)
                nn_action = nn_action_dict['action']
                action = convert_action(nn_action, steer_range=cfg.envs.steer_range, speed_range=cfg.envs.speed_range)
            else:
                action = planner.plan(obs, id=i)
            actions.append(action)

        # 環境のステップ
        next_obs, reward, terminated, truncated, info = env.step(np.array(actions))
        done = terminated or truncated

        # スキャンの更新
        next_scan = convert_scan(next_obs['scans'][0], cfg.envs.max_beam_range)
        scan_buffer.add_scan(next_scan)

        total_reward += reward

        if done:
            break

        if cfg.render:
            env.render(cfg.render_mode)

        obs = next_obs

    print(f"Evaluation completed. Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
