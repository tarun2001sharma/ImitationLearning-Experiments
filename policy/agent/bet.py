import torch
from torch import nn
import torchvision.transforms as T
from torch.nn import functional as F
import torch.distributions as D

import utils
from agent.networks.encoder import Encoder
from agent.networks.kmeans_discretizer import KMeansDiscretizer

class FocalLoss(nn.Module):
    
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        """
		Args:
			input: (N, B), where B = number of bins
			target: (N, )
		"""
        logpt = F.log_softmax(input, dim=-1)  # shape (N, nbins)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)  # pick log prob of correct bin
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim, nbins):
        super().__init__()

        self._output_dim = action_shape[0]

        # Common trunk
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

		# TODO: Define the policy network
		# Hint: There must be a common trunk followed by two heads - one for binning and one for offsets
        # Two heads:
        # 1) bin_head: outputs nbins logits for the entire action (not per-dim)
        # 2) offset_head: outputs a continuous offset for each action dim
        self.bin_head = nn.Linear(hidden_dim, nbins)
        self.offset_head = nn.Linear(hidden_dim, self._output_dim)

        self.apply(utils.weight_init)

    def forward(self, obs, std, cluster_centers=None):
		# TODO: Implement the forward pass, return bin_logits, and offset
        features = self.trunk(obs)
        bin_logits = self.bin_head(features)       # (batch_size, nbins)
        offset = self.offset_head(features)        # (batch_size, action_dim)
        return bin_logits, offset

class Agent:
    def __init__(
        self, 
        obs_shape, 
        action_shape, 
        device, 
        lr, 
        hidden_dim, 
        stddev_schedule,
        stddev_clip, 
        use_tb, 
        obs_type, 
        nbins, 
        kmeans_iters, 
        offset_weight,
        offset_loss_weight
    ):

        self.device = device
        self.lr = lr
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.use_encoder = (obs_type == 'pixels')
        self.nbins = nbins
        self.kmeans_iters = kmeans_iters
        self.offset_weight = offset_weight
        self.offset_loss_weight = offset_loss_weight
        
        self._act_dim = action_shape[0]

        # KMeans-based discretizer
        self.discretizer = KMeansDiscretizer(num_bins=self.nbins, kmeans_iters=self.kmeans_iters)
        self.cluster_centers = None  # will be set after fitting

       	# TODO: Define the encoder (for pixels)
        if self.use_encoder:
            self.encoder = Encoder(obs_shape).to(device)
            repr_dim = self.encoder.repr_dim
        else:
			# TODO: Define the representation dimension for non-pixel observations
            # if features, e.g. obs_dim=4
            repr_dim = obs_shape[0]

		# TODO: Define the actor
        self.actor = Actor(repr_dim, action_shape, hidden_dim, nbins).to(device)

        # Loss 
        self.criterion = FocalLoss(gamma=2.0)

        # TODO: Define optimizers
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        else:
            self.encoder_opt = None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        # Data augmentation (for pixel observations)
        if self.use_encoder:
            self.aug = utils.RandomShiftsAug(pad=4)

        self.train()

    def __repr__(self):
        return "bet"
    
    def train(self, training=True):
        self.training = training
        if training:
            if self.use_encoder:
                self.encoder.train(training)
            self.actor.train(training)
        else:
            if self.use_encoder:
                self.encoder.eval()
            self.actor.eval()
	
    def compute_action_bins(self, actions):
        actions = torch.as_tensor(actions, device=self.device).float()
        self.discretizer.fit(actions)
        # store cluster centers for later usage (shape: (nbins, action_dim))
        self.cluster_centers = self.discretizer.bin_centers.float().to(self.device)
		
    def find_closest_cluster(self, actions) -> torch.Tensor:
        # TODO: Return the index of closest cluster center for each action in actions
		# Return shape: (N, )
        """
        For each action in 'actions', return the index of the closest cluster center.
        actions: (N, action_dim)
        Returns: (N,) of cluster indices
        """
        # shape: (N, 1, action_dim) - (1, nbins, action_dim)
        diff = actions.unsqueeze(1) - self.cluster_centers.unsqueeze(0)  # (N, nbins, action_dim)
        dist = (diff ** 2).sum(dim=-1)  # (N, nbins)
        closest_cluster_center = dist.argmin(dim=1)  # (N,)
        return closest_cluster_center

    def act(self, obs, goal, step):
		# TODO: Obtain bin_logits and offset from the actor

        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0)

        if self.use_encoder:
            obs = self.encoder(obs)

		# TODO: Compute base action (Hint: Use the bin_logits)
        # Forward pass: get bin logits + offset
        bin_logits, offset = self.actor(obs, None, cluster_centers=self.cluster_centers)

        # base_action is the cluster center for the argmax bin
        bin_probs = F.softmax(bin_logits, dim=-1)  # (1, nbins)
        bin_idx = bin_probs.argmax(dim=-1)         # (1,)
        base_action = self.cluster_centers[bin_idx]  # (1, action_dim)

		# TODO: Compute base action (Hint: Use the bin_logits)
        # final action = base_action + offset_weight * offset
        action = base_action + self.offset_weight * offset
        return action.cpu().numpy()[0]

    def update(self, expert_replay_iter, step):
        metrics = dict()

        batch = next(expert_replay_iter)
        obs, action, goal = utils.to_torch(batch, self.device)
        obs, action, goal = obs.float(), action.float(), goal.float()
		
        # augment + encode
        if self.use_encoder:
            obs = self.aug(obs)
            obs = self.encoder(obs)

		# TODO: Compute bin_logits and offset from the actor
        bin_logits, offset = self.actor(obs, None, cluster_centers=self.cluster_centers)

		# TODO: Compute discrete loss on bins and offset loss
        # 1) Discrete classification: which bin is closest to the expert action?
        closest_bin_idx = self.find_closest_cluster(action)  # (batch_size,)
        discrete_loss = self.criterion(bin_logits, closest_bin_idx)

        # 2) Offset regression: the difference between expert action and the chosen bin center
        # shape (batch_size, action_dim)
        base_action = self.cluster_centers[closest_bin_idx]
        target_offset = action - base_action  # how far the expert action is from the bin center
        offset_loss = F.mse_loss(offset, target_offset)

        # actor loss (combine)
        actor_loss = discrete_loss + self.offset_loss_weight * offset_loss

        if self.use_encoder:
            self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()

        actor_loss.backward()
        
        if self.use_encoder:
            self.encoder_opt.step()
        self.actor_opt.step()

        
		# TODO: Update the actor (and encoder for pixels)
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['discrete_loss'] = discrete_loss.item()
            metrics['offset_loss'] = offset_loss.item() * self.offset_loss_weight
            metrics['logits_entropy'] = D.Categorical(logits=bin_logits).entropy().mean().item()

        return metrics

    def save_snapshot(self):
        keys_to_save = ['actor', 'cluster_centers']
        if self.use_encoder:
            keys_to_save += ['encoder']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v

        # Recreate the optimizers after loading
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
