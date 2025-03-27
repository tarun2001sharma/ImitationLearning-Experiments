import torch
from torch import nn
import torchvision.transforms as T

import utils
from agent.networks.encoder import Encoder

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self._output_dim = action_shape[0]
        
        # -----------------------------------------------------------------------------
        # Policy network: a two-layer MLP that outputs raw logits,
        # which we pass through tanh in forward().
        # -----------------------------------------------------------------------------
        self.policy = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self._output_dim)  # raw output
        )
        # -----------------------------------------------------------------------------

        self.apply(utils.weight_init)

    def forward(self, obs, std):
       
        mu_raw = self.policy(obs)
        mu = torch.tanh(mu_raw)

        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class BCAgent:
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
        obs_type
    ):
       
        self.device = device
        self.lr = lr
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.use_encoder = (obs_type == 'pixels')
        
        # Actor parameters
        self._act_dim = action_shape[0]

        # Models
        if self.use_encoder:
            self.encoder = Encoder(obs_shape).to(device)
            repr_dim = self.encoder.repr_dim
        else:
            # If observations are a simple vector, obs_shape is (obs_dim,)
            repr_dim = obs_shape[0]

        self.actor = Actor(repr_dim, action_shape, hidden_dim).to(device)

        # -----------------------------------------------------------------------------
        # Define optimizers
        # -----------------------------------------------------------------------------
        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        else:
            self.encoder_opt = None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        # -----------------------------------------------------------------------------

        # Data augmentation 
        if self.use_encoder:
            self.aug = utils.RandomShiftsAug(pad=4)

        self.train()

    def __repr__(self):
        return "bc"
    
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

    def act(self, obs, goal, step):
        # Convert to tensor and add batch dimension
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0)
        
        stddev = utils.schedule(self.stddev_schedule, step)
        stddev = max(0.0, min(stddev, self.stddev_clip))  # clip std

        if self.use_encoder:
            obs = self.encoder(obs)
            
		# TODO: Compute action using the actor (and the encoder if pixels are used)
        dist_action = self.actor(obs, stddev)
        # Use the mean of the distribution as the action
        action = dist_action.mean
        return action.cpu().numpy()[0]

    def update(self, expert_replay_iter, step):
        metrics = dict()

        batch = next(expert_replay_iter)
        obs, action, goal = utils.to_torch(batch, self.device)
        obs, action, goal = obs.float(), action.float(), goal.float()
        
        # Augment 
        if self.use_encoder:
			# TODO: Augment the observations and encode them (for pixels)
            obs = self.aug(obs)
            obs = self.encoder(obs)

        stddev = utils.schedule(self.stddev_schedule, step)
        stddev = max(0.0, min(stddev, self.stddev_clip))
        
		# TODO: Compute the actor loss using log_prob on output of the actor
        dist = self.actor(obs, stddev)
        # Negative log-likelihood of expert action => BC loss
        log_prob = dist.log_prob(action)  # shape [batch_size, action_dim]
        actor_loss = -log_prob.sum(dim=-1).mean()  # sum across action dims, then mean

        # Zero grads
        if self.use_encoder:
            self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()

        # Backprop
        actor_loss.backward()

		# TODO: Update the actor (and encoder for pixels)		
        if self.use_encoder:
            self.encoder_opt.step()
        self.actor_opt.step()

        # Log
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()

        return metrics

    def save_snapshot(self):
        keys_to_save = ['actor']
        if self.use_encoder:
            keys_to_save += ['encoder']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        return payload

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
