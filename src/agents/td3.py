import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.model import get_seterministic_actor, get_critic
from src.utils.helper import soft_update


class TD3:
    def __init__(self,
                 actor_cfg,
                 critic_cfg,
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 device="cpu"):
        
        self.device = device
        
        # Actor, Critic の初期化
        self.actor = get_seterministic_actor(actor_cfg=actor_cfg).to(device)
        self.actor_target = get_seterministic_actor(actor_cfg=actor_cfg).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = get_critic(critic_cfg=critic_cfg).to(device)
        self.critic2 = get_critic(critic_cfg=critic_cfg).to(device)
        self.critic1_target = get_critic(critic_cfg=critic_cfg).to(device)
        self.critic2_target = get_critic(critic_cfg=critic_cfg).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizer の設定
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_cfg.lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_cfg.lr
        )

        # ハイパーパラメータ
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_counter = 0

    def select_action(self, state, evaluate=False, noise=0.1):
        """
        アクションの選択
        :param state: 現在の状態
        :param evaluate: 評価モードかどうか
        :param noise: 探索時のノイズ
        :return: 選択されたアクション
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if not evaluate and noise != 0:
            # 学習時はノイズを付与して探索を促進
            action = action + np.random.normal(0, noise, size=action.shape)
        
        # アクションのクリッピング
        return np.clip(action, -1, 1)

    def update(self, replay_buffer, batch_size=256):
        """ネットワークの更新処理"""
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Critic の更新
        self._update_critic(state, action, reward, next_state, done)
        
        # ポリシーの遅延更新
        if self.update_counter % self.policy_delay == 0:
            self._update_actor(state)
            
            # Soft Update の実行
            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic1, self.critic1_target, self.tau)
            soft_update(self.critic2, self.critic2_target, self.tau)
        
        # カウンターの更新
        self.update_counter += 1

    def _update_critic(self, state, action, reward, next_state, done):
        """Critic の更新処理"""
        with torch.no_grad():
            # ポリシーノイズの追加
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            # ターゲット Q 値の計算
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * torch.min(target_q1, target_q2)

        # 現在の Q 値を取得
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # 損失計算
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        # オプティマイザの更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def _update_actor(self, state):
        """Actor の更新処理"""
        actor_loss = -self.critic1(state, self.actor(state)).mean()

        # オプティマイザの更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self, path):
        """モデルの保存"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path):
        """モデルのロード"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
