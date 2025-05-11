import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.model import get_ppo_actor, get_ppo_critic

class PPO:
    def __init__(self,
                 actor_cfg,
                 critic_cfg,
                 gamma=0.99,
                 tau=0.005,
                 epsilon=0.2,
                 update_epochs=10,
                 device="cpu"):
        self.device = device
        
        # Actor, Critic の初期化
        self.actor = get_ppo_actor(actor_cfg=actor_cfg).to(device)
        self.critic = get_ppo_critic(critic_cfg=critic_cfg).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_cfg.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), critic_cfg.lr)

        # ハイパーパラメータ
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.update_epochs = update_epochs

    def select_action(self, state, evaluate=False):
        """アクションの選択"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor(state)
            action = torch.tanh(action)  # 学習時もクリッピング
        return {
            "action": action.cpu().numpy()[0],
            "log_prob": log_prob.cpu().numpy()[0] if log_prob is not None else None
        }

    def compute_gae(self, rewards, values, next_values, dones):
        """
        Generalized Advantage Estimation (GAE) の計算
        """
        gae = 0
        advantage = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1 - dones[step]) * gae
            advantage.insert(0, gae)
        return torch.tensor(advantage, dtype=torch.float32).to(self.device)

    def update(self, buffer, mini_batch_size=256):
        """エピソード終了後のネットワークの更新処理"""
        # バッファから全てのデータを取り出す
        state, action, reward, next_state, done, old_log_prob = buffer.get()

        # 価値の計算
        with torch.no_grad():
            values = self.critic(state)
            next_values = self.critic(next_state)

        # Advantage の計算
        advantage = self.compute_gae(reward, values, next_values, done)

        # --- ミニバッチに分割して学習 ---
        dataset_size = state.size(0)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        # バッチごとに PPO の更新
        for start in range(0, dataset_size, mini_batch_size):
            end = min(start + mini_batch_size, dataset_size)  # サイズを超えた場合、最後まで取る
            mb_indices = indices[start:end]

            mb_states = state[mb_indices]
            mb_actions = action[mb_indices]
            mb_old_log_probs = old_log_prob[mb_indices]
            mb_advantage = advantage[mb_indices]
            mb_rewards = reward[mb_indices]
            mb_next_states = next_state[mb_indices]
            mb_dones = done[mb_indices]

            # Actor と Critic の更新処理
            a_info = self._update_actor(mb_states, mb_actions, mb_old_log_probs, mb_advantage)
            c_info = self._update_critic(mb_states, mb_rewards, mb_next_states, mb_dones)

        return {**a_info, **c_info}

    def _update_actor(self, state, action, old_log_prob, advantage):
        """Actor の更新処理"""
        for _ in range(self.update_epochs):
            # 新しいポリシーでの log_prob を取得
            _, log_prob = self.actor(state)
            ratio = (log_prob - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
            loss = -torch.min(surr1, surr2).mean()

            # オプティマイザの更新
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        
        return {'actor_loss': loss.item()}

    def _update_critic(self, state, reward, next_state, done):
        """Critic の更新処理"""
        with torch.no_grad():
            target_value = reward + (1 - done) * self.gamma * self.critic(next_state)
        
        value = self.critic(state)
        loss = nn.MSELoss()(value, target_value)

        # オプティマイザの更新
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        return {'critic_loss': loss.item()}

    def save(self, path):
        """モデルの保存"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path):
        """モデルのロード"""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
