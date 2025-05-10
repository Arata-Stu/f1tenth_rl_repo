from omegaconf import DictConfig

from .actor import Gaussian1dConvPolicy, GaussianPolicy, Deterministic1dConvPolicy, DeterministicPolicy
from .critic import Double1dConvCritic, DoubleCritic

def get_stochastic_actor(actor_cfg: DictConfig):
    """
    Actorの取得
    """
    actor_name = actor_cfg.name
    if actor_name == "1dconv":
        return Gaussian1dConvPolicy(state_dim=actor_cfg.state_dim,
                                    action_dim=actor_cfg.action_dim,
                                    hidden_dim=actor_cfg.hidden_dim)
    elif actor_name == "mlp":
        return GaussianPolicy(state_dim=actor_cfg.state_dim,
                              action_dim=actor_cfg.action_dim,
                              hidden_dim=actor_cfg.hidden_dim)
    else:
        raise ValueError(f"Unknown actor name: {actor_name}")
    
def get_seterministic_actor(actor_cfg: DictConfig):
    """
    Actorの取得
    """
    actor_name = actor_cfg.name
    if actor_name == "1dconv":
        return Deterministic1dConvPolicy(state_dim=actor_cfg.state_dim,
                                    action_dim=actor_cfg.action_dim,
                                    hidden_dim=actor_cfg.hidden_dim)
    elif actor_name == "mlp":
        return DeterministicPolicy(state_dim=actor_cfg.state_dim,
                              action_dim=actor_cfg.action_dim,
                              hidden_dim=actor_cfg.hidden_dim)
    else:
        raise ValueError(f"Unknown actor name: {actor_name}")    
    
def get_critic(critic_cfg: DictConfig):
    """
    Criticの取得
    """
    critic_name = critic_cfg.name
    if critic_name == "1dconv":
        return Double1dConvCritic(state_dim=critic_cfg.state_dim,
                                  action_dim=critic_cfg.action_dim,
                                  hidden_dim=critic_cfg.hidden_dim,
                                  tau=critic_cfg.tau)
    elif critic_name == "mlp":
        return DoubleCritic(state_dim=critic_cfg.state_dim,
                            action_dim=critic_cfg.action_dim,
                            hidden_dim=critic_cfg.hidden_dim,
                            tau=critic_cfg.tau)
    else:
        raise ValueError(f"Unknown critic name: {critic_name}")