from omegaconf import DictConfig

from src.agents.sac import SAC
from src.agents.td3 import TD3
def get_agent(agent_cfg: DictConfig, device: str):
    """
    エージェントの初期化
    """
    agent_name = agent_cfg.name
    if agent_name == "sac":
        return SAC(
            actor_cfg=agent_cfg.actor,
            critic_cfg=agent_cfg.critic,
            alpha_lr=agent_cfg.alpha_lr,
            gamma=agent_cfg.gamma,
            tau=agent_cfg.tau,
            target_entropy=agent_cfg.target_entropy,
            device=device
        )
    elif agent_name == "td3":
        return TD3(
            actor_cfg=agent_cfg.actor,
            critic_cfg=agent_cfg.critic,
            gamma=agent_cfg.gamma,
            tau=agent_cfg.tau,
            policy_noise=agent_cfg.policy_noise,
            noise_clip=agent_cfg.noise_clip,
            policy_delay=agent_cfg.policy_delay,
            device=device
        )
    else:
        raise ValueError(f"Unknown agent name: {agent_name}")
    