import numpy as np
import torch
from typing import Union, Tuple

class OnPolicyBuffer:
    """
    PPO 用のオンポリシーリプレイバッファ。
    エピソードごとに収集し、学習後にリセットされる。
    """
    def __init__(self,
                 state_shape: Union[int, Tuple[int, ...]],
                 action_dim: int,
                 buffer_size: int,
                 device: str = "cpu"):
        """
        Args:
            state_shape (int or tuple): 状態ベクトルの形状
            action_dim (int): 行動ベクトルの次元
            buffer_size (int): バッファの最大サイズ（1エピソードのステップ数）
            device (str): サンプリング時に返す Tensor のデバイス
        """
        self.device = device
        self.state_shape = (state_shape,) if isinstance(state_shape, int) else state_shape
        self.action_dim = action_dim
        self.buffer_size = buffer_size

        # バッファの初期化
        self.reset()

    def reset(self):
        """バッファを空にしてすべての配列を再確保"""
        self.ptr = 0
        self.full = False
        self.states = np.zeros((self.buffer_size, *self.state_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, *self.state_shape), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, 1), dtype=np.float32)

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """入力を float32 の np.ndarray に変換"""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)
        return x.astype(np.float32)

    def add(self, state, action, reward, next_state, done, log_prob):
        """
        1 タイムステップ分のデータを追加。
        """
        idx = self.ptr

        self.states[idx] = self._to_numpy(state)
        self.actions[idx] = self._to_numpy(action)
        self.rewards[idx] = self._to_numpy(reward)
        self.next_states[idx] = self._to_numpy(next_state)
        self.dones[idx] = self._to_numpy(done)
        self.log_probs[idx] = self._to_numpy(log_prob)

        # ポインタ更新
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
            self.ptr = 0

    def get(self):
        """
        バッファから全てのデータを取得し、Tensor で返す。
        バッファはリセットされる。
        """
        assert self.full, "バッファが満たされていません。すべてのステップが収集されていません。"

        data = (
            torch.tensor(self.states, dtype=torch.float32).to(self.device),
            torch.tensor(self.actions, dtype=torch.float32).to(self.device),
            torch.tensor(self.rewards, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_states, dtype=torch.float32).to(self.device),
            torch.tensor(self.dones, dtype=torch.float32).to(self.device),
            torch.tensor(self.log_probs, dtype=torch.float32).to(self.device)
        )

        # 使用後はバッファをリセット
        self.reset()
        return data

    def __len__(self):
        return self.buffer_size if self.full else self.ptr
