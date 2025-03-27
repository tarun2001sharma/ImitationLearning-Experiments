# import torch
# from torch import nn
# import torchvision.transforms as T

# import utils
# from agent.networks.encoder import Encoder

# class Actor(nn.Module):
#     def __init__(self, repr_dim, action_shape, hidden_dim):
#         super().__init__()

#         self._output_dim = action_shape[0]
        
# 		# TODO: Define the policy network
#         # -------------------------------------------------------------------------
#         # two-layer MLP that takes [obs, goal] concatenated
#         # (so repr_dim should include the goal dimension)
#         self.policy = nn.Sequential(
#             nn.Linear(repr_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, self._output_dim)
#         )

#         self.apply(utils.weight_init)

#     def forward(self, obs, std):
# 		# TODO: Implement the forward pass
#         mu_raw = self.policy(obs)
#         mu = torch.tanh(mu_raw)

#         # Construct a diagonal truncated normal distribution
#         std = torch.ones_like(mu) * std
#         dist = utils.TruncatedNormal(mu, std)
#         return dist


# class BCAgent:
#     def __init__(
#         self, 
#         obs_shape, 
#         action_shape, 
#         device, 
#         lr, 
#         hidden_dim, 
#         stddev_schedule, 
#         stddev_clip, 
#         use_tb, 
#         obs_type
#     ):

#         self.device = device
#         self.lr = lr
#         self.stddev_schedule = stddev_schedule
#         self.stddev_clip = stddev_clip
#         self.use_tb = use_tb
#         self.use_encoder = (obs_type == 'pixels')
        
#         # Actor parameters
#         self._act_dim = action_shape[0]

# 		# TODO: Define the encoder (for pixels) and define the representation dimension for non-pixel observations
#         # -------------------------------------------------------------------------
#         # Define the encoder otherwise, define repr_dim
#         # to include both observation + goal dims.
#         if self.use_encoder:
#             self.encoder = Encoder(obs_shape).to(device)
#             # If the encoder outputs self.encoder.repr_dim, add 2 for the goal
#             repr_dim = self.encoder.repr_dim + 2  
#         else:
# 			# TODO: Define the representation dimension for non-pixel observations
#             # For a feature-based environment, obs_shape[0] is the obs dimension
#             # Add 2 for the goal dimension if the goal is 2D
#             repr_dim = obs_shape[0] + 2

# 		# TODO: Define the actor
#         # -------------------------------------------------------------------------
#         self.actor = Actor(repr_dim, action_shape, hidden_dim).to(device)
		
# 		# TODO: Define optimizers
#         # -------------------------------------------------------------------------
#         if self.use_encoder:
#             self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
#         else:
#             self.encoder_opt = None

#         self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

#         # Data augmentation
#         if self.use_encoder:
#             self.aug = utils.RandomShiftsAug(pad=4)

#         self.train()

#     def __repr__(self):
#         return "gcbc"
    
#     def train(self, training=True):
#         self.training = training
#         if training:
#             if self.use_encoder:
#                 self.encoder.train(training)
#             self.actor.train(training)
#         else:
#             if self.use_encoder:
#                 self.encoder.eval()
#             self.actor.eval()

#     def act(self, obs, goal, step):
#         # Convert to tensors and add batch dimension
#         obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
#         goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0)

#         stddev = utils.schedule(self.stddev_schedule, step)
#         stddev = max(0.0, min(stddev, self.stddev_clip))  # clip

#         # TODO: Compute action using the actor (and the encoder if pixels are used)
#         # -------------------------------------------------------------------------
#         if self.use_encoder:
#             obs = self.encoder(obs)

#         # Concatenate [obs, goal] along the last dimension
#         combined = torch.cat([obs, goal], dim=-1)

#         dist_action = self.actor(combined, stddev)

#         action = dist_action.mean
#         return action.cpu().numpy()[0]

#     def update(self, expert_replay_iter, step):
#         metrics = dict()

#         batch = next(expert_replay_iter)
#         obs, action, goal = utils.to_torch(batch, self.device)
#         obs, action, goal = obs.float(), action.float(), goal.float()

#         if self.use_encoder:
# 			# TODO: Augment the observations and encode them (for pixels)
#             obs = self.aug(obs)
#             obs = self.encoder(obs)

#         # Concatenate [obs, goal]
#         combined = torch.cat([obs, goal], dim=-1)

#         stddev = utils.schedule(self.stddev_schedule, step)
#         stddev = max(0.0, min(stddev, self.stddev_clip))

#         # Compute distribution
#         dist = self.actor(combined, stddev)

#         # TODO: Compute the actor loss using log_prob on output of the actor and 
#         # Update the actor (and encoder for pixels)	
#         log_prob = dist.log_prob(action)  # shape: [batch_size, action_dim]
#         actor_loss = -log_prob.sum(dim=-1).mean()  # sum over action dims, then average

#         # Zero grads
#         if self.use_encoder:
#             self.encoder_opt.zero_grad()
#         self.actor_opt.zero_grad()

#         # Backprop
#         actor_loss.backward()

#         # Step
#         if self.use_encoder:
#             self.encoder_opt.step()
#         self.actor_opt.step()

#         # Log
#         if self.use_tb:
#             metrics['actor_loss'] = actor_loss.item()
        
#         return metrics

#     def save_snapshot(self):
#         keys_to_save = ['actor']
#         if self.use_encoder:
#             keys_to_save += ['encoder']
#         payload = {k: self.__dict__[k] for k in keys_to_save}
#         return payload

#     def load_snapshot(self, payload):
#         for k, v in payload.items():
#             self.__dict__[k] = v

import torch
from torch import nn
import torchvision.transforms as T

import utils
from agent.networks.encoder import Encoder

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self._output_dim = action_shape[0]
        
        self.policy = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim), # First hidden layer   
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Second hidden layer    
            nn.ReLU(),
            nn.Linear(hidden_dim, self._output_dim) # Output layer producing action values  
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        action_raw = self.policy(obs)
        
        action_mean = torch.tanh(action_raw)
        
        action_std = torch.ones_like(action_mean) * std
        
        action_dist = utils.TruncatedNormal(action_mean, action_std)

        return action_dist

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
        
        self._act_dim = action_shape[0]

        if self.use_encoder:
            self.encoder = Encoder(obs_shape).to(device)
            repr_dim = self.encoder.repr_dim + 2  
        else:
            repr_dim = obs_shape[0] + 2

        self.actor = Actor(repr_dim, action_shape, hidden_dim).to(device)

        if self.use_encoder:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        else:
            self.encoder_opt = None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        if self.use_encoder:
            self.aug = utils.RandomShiftsAug(pad=4)

        self.train()

    def __repr__(self):
        return "gcbc"
    
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
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).float().unsqueeze(0)

        stddev = utils.schedule(self.stddev_schedule, step)
        stddev = max(0.0, min(stddev, self.stddev_clip))

        if self.use_encoder:
            obs = self.encoder(obs)

        merged = torch.cat([obs, goal], dim=-1)

        dist_action = self.actor(merged, stddev)

        action = dist_action.mean

        return action.cpu().numpy()[0]

    def update(self, expert_replay_iter, step):
        metrics = dict()

        batch = next(expert_replay_iter)
        obs, action, goal = utils.to_torch(batch, self.device)
        obs, action, goal = obs.float(), action.float(), goal.float()

        if self.use_encoder:
            obs = self.aug(obs)
            obs = self.encoder(obs)

        merged = torch.cat([obs, goal], dim=-1)

        stddev = utils.schedule(self.stddev_schedule, step)
        #stddev = max(0.0, min(stddev, self.stddev_clip))

        dist = self.actor(merged, stddev)

        log_prob = dist.log_prob(action)
        actor_loss = -log_prob.sum(dim=-1).mean()

        if self.use_encoder:
            self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()
        actor_loss.backward()

        if self.use_encoder:
            self.encoder_opt.step()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
        
        return metrics

    def save_snapshot(self):
        keys_to_save = ['actor']
        if self.use_encoder:
            keys_to_save.append('encoder')
        return {k: self.__dict__[k] for k in keys_to_save}

    def load_snapshot(self, payload):
        for k, v in payload.items():
            self.__dict__[k] = v
