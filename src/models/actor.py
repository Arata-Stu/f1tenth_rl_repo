import torch
import torch.nn as nn

from .backbone import TinyLidarBackbone

class GaussianPolicy(nn.Module):
    """
    ガウス政策ネットワーク (連続行動空間)
    出力は平均と対数分散を返し、sample()でtanh補正後の行動とログ確率を返す。
    """
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)
    
class Gaussian1dConvPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.lidar_backbone = TinyLidarBackbone(input_dim=state_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.net = nn.Sequential(
            nn.Linear(self.lidar_backbone.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.lidar_backbone(state)
        x = self.net(features)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)


class DeterministicPolicy(nn.Module):
    """
    TD3用のDeterministic Policy Network
    状態を受け取り、直接アクションを推定する。
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        """
        入力:
            state: [batch_size, state_dim]
        出力:
            action: [-1, 1] の範囲に正規化されたアクション
        """
        return self.net(state)


class Deterministic1dConvPolicy(nn.Module):
    """
    TD3用のConvベースのDeterministic Policy Network
    LiDARの入力をTinyLidarBackboneで特徴抽出し、アクションを推定する。
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.lidar_backbone = TinyLidarBackbone(input_dim=state_dim)
        self.net = nn.Sequential(
            nn.Linear(self.lidar_backbone.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        """
        入力:
            state: [batch_size, state_dim]
        出力:
            action: [-1, 1] の範囲に正規化されたアクション
        """
        features = self.lidar_backbone(state)
        return self.net(features)
    


class PPOGaussianActor(nn.Module):
    """
    PPO 向けのガウス政策ネットワーク (連続行動空間)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, log_std_min=-20, log_std_max=2):
        super(PPOGaussianActor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # ネットワーク定義
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 平均と標準偏差のパラメータ
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        状態を入力にして、平均 (mean) と対数標準偏差 (log_std) を出力
        """
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """
        ガウス分布からサンプリングしてアクションとログ確率を返す
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # 再パラメータ化トリック
        z = normal.rsample()
        action = torch.tanh(z)

        # ログ確率の計算
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob
    

class PPOGaussian1dConvActor(nn.Module):
    """
    PPO 向けのガウス政策ネットワーク with TinyLidarBackbone
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, log_std_min=-20, log_std_max=2):
        super().__init__()
        # TinyLidarBackbone の初期化
        self.lidar_backbone = TinyLidarBackbone(input_dim=state_dim)
        
        # 標準偏差の範囲制限
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # ネットワーク定義
        self.net = nn.Sequential(
            nn.Linear(self.lidar_backbone.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 平均と標準偏差のパラメータ
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        状態を入力にして、TinyLidarBackbone を通し、
        特徴量を取得した後、平均 (mean) と対数標準偏差 (log_std) を出力
        """
        features = self.lidar_backbone(state)
        x = self.net(features)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """
        ガウス分布からサンプリングしてアクションとログ確率を返す
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        # 再パラメータ化トリック
        z = normal.rsample()
        action = torch.tanh(z)

        # ログ確率の計算
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob