# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RkKPo_EsfOzZWfaoog00hfMJ9P6OWQlS
"""

import torch
from torch import nn, cuda, optim
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import time
import datetime
from argparse import ArgumentParser as ap

dvc = torch.device('cuda' if cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.conv = nn.Conv1d(
            input_size,
            hidden_size,
            kernel_size=1
        )

    def forward(self, input):
        """forward."""
        output = self.conv(input)
        return output # (batch, hidden_size, seq_len)

class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super().__init__()
        # W processes features from static decoder elements
        self.v = nn.Parameter(
            torch.zeros(
                (1, 1, hidden_size),
                device=dvc,
                requires_grad=True
            )
        )
        self.W = nn.Parameter(
            torch.zeros(
                (1, hidden_size, 3 * hidden_size),
                device=dvc,
                requires_grad=True
            )
        )

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        """forward."""
        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = F.softmax(torch.bmm(v, torch.tanh(torch.bmm(W, hidden))),dim=2)
        return attns # (batch, seq_len)

class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(
            torch.zeros(
                (1, 1, hidden_size),
                device=dvc,
                requires_grad=True
            )
        )
        self.W = nn.Parameter(
            torch.zeros(
                (1, hidden_size, 2 * hidden_size),
                device=dvc,
                requires_grad=True
            )
        )
        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_attn = Attention(hidden_size)
        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        """forward"""
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)
        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 
        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1)) # (B, 1, num_feats)
        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1) # (B, num_feats, seq_len)
        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)
        return probs, last_hh

class DRL4TSP(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatibility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidelines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(
        self, static_size, dynamic_size, hidden_size,
        update_fn=None, mask_fn=None, num_layers=1, dropout=0.0
    ):
        super().__init__()
        if dynamic_size < 1:
            raise ValueError(
                ':param dynamic_size: must be > 0, even if the problem has no dynamic elements'
            )
        self.update_fn = update_fn
        self.mask_fn = mask_fn
        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros(
            (1, static_size, 1),
            requires_grad=True,
            device=dvc
        )

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        batch_size, input_size, sequence_size = static.size()
        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)
        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(
            batch_size,
            sequence_size,
            device=dvc
        )
        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 1000
        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        for _ in range(max_steps):
            if not mask.byte().any():
                break
            # ... but compute a hidden rep for each element added to sequence
            decoder_hidden = self.decoder(decoder_input)
            probs, last_hh = self.pointer(
                static_hidden,
                dynamic_hidden,
                decoder_hidden,
                last_hh
            )
            probs = F.softmax(probs + mask.log(), dim=1)
            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)
                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1) # Greedy
                logp = prob.log()
            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)
                # Since we compute the VRP in mini-batches, some tours may have
                # number of stops. We force the vehicles to remain at the depot 
                # in these cases, and logp := 0
                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)
            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))
            decoder_input = torch.gather(
                static, 2,
                ptr.view(-1, 1, 1).expand(-1, input_size, 1)
            ).detach()
        tour_idx = torch.cat(tour_idx, dim=1) # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1) # (batch_size, seq_len)
        return tour_idx, tour_logp

class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super().__init__()
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        """forward"""
        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        hidden = torch.cat((static_hidden, dynamic_hidden), 1)
        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output

class Critic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, hidden_size):
        super().__init__()
        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(1, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        """forward"""
        output = F.relu(self.fc1(input.unsqueeze(1)))
        output = F.relu(self.fc2(output)).squeeze(2)
        output = self.fc3(output).sum(dim=2)
        return output

"""Defines the main task for the VRP.

The VRP is defined by the following traits:
    1. Each city has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a capacity (depends on problem), the must visit all cities
    3. When the vehicle load is 0, it __must__ return to the depot to refill
"""

class VehicleRoutingDataset(Dataset):

    def __init__(self, num_samples, input_size, max_load=20, max_demand=9,seed=None):
        super().__init__()
        if max_load < max_demand:
            raise ValueError(
                ':param max_load: must be > max_demand'
            )
        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand
        # Depot location will be the first node in each
        locations = torch.rand((num_samples, 2, input_size + 1))
        self.static = locations
        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.)
        # All states will have their own intrinsic demand in [1, max_demand), 
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30, 
        # demands will be scaled to the range (0, 3)
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load)
        demands[:, 0, 0] = 0 # depot starts with a demand of 0
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """
        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0] # (batch_size, seq_len)
        demands = dynamic.data[:, 1] # (batch_size, seq_len)
        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.0
        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0) * demands.lt(loads)
        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)
        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if (torch.logical_not(repeat_home)).any():
            new_mask[torch.logical_not(repeat_home).nonzero(), 0] = 0.0
        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()
        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.
        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (load, demand) dataset values."""
        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)
        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()
        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))
        # Across the mini-batch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():
            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)
            # Broadcast the load to all nodes, but update demand separately
            visit_idx = visit.nonzero().squeeze()
            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1.0 + new_load[visit_idx].view(-1)
        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.0
            all_demands[depot.nonzero().squeeze(), 0] = 0.0
        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        return tensor.data.to(device=dynamic.device).clone().detach().requires_grad_(True)

class Simulate:

    def __init__(self, static, tour_indices):
        self.static = static
        self.tour_indices = tour_indices

    def reward(self):
        """Euclidean distance between all cities / nodes given by tour_indices"""
        # Convert the indices back into a tour
        idx = self.tour_indices.unsqueeze(1).expand(-1, self.static.size(1), -1)
        tour = torch.gather(self.static.data, 2, idx).permute(0, 2, 1)
        # Ensure we're always returning to the depot - note the extra concat
        # won't add any extra loss, as the euclidean distance between consecutive
        # points is 0
        start = self.static.data[:, :, 0].unsqueeze(1)
        y = torch.cat((start, tour, start), dim=1)
        # Euclidean distance between each consecutive point
        tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
        return tour_len.sum(1)

    def render(self, save_path):
        """Plots the found solution."""
        plt.close('all')
        num_plots = 3 if int(np.sqrt(len(self.tour_indices))) >= 3 else 1
        _, axes = plt.subplots(nrows=num_plots,ncols=num_plots,sharex='col',sharey='row')
        if num_plots == 1:
            axes = [[axes]]
        axes = [a for ax in axes for a in ax]
        for i, ax in enumerate(axes):
            # Convert the indices back into a tour
            idx = self.tour_indices[i]
            if len(idx.size()) == 1:
                idx = idx.unsqueeze(0)
            idx = idx.expand(self.static.size(1), -1)
            data = torch.gather(self.static[i].data, 1, idx).cpu().numpy()
            start = self.static[i, :, 0].cpu().data.numpy()
            x = np.hstack((start[0], data[0], start[0]))
            y = np.hstack((start[1], data[1], start[1]))
            # Assign each sub-tour a different color & label in order traveled
            idx = np.hstack((0, self.tour_indices[i].cpu().numpy().flatten(), 0))
            where = np.where(idx == 0)[0]
            for j in range(len(where) - 1):
                low = where[j]
                high = where[j + 1]
                if low + 1 == high:
                    continue
                ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)
            ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
            ax.scatter(x, y, s=4, c='r', zorder=2)
            ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)

    def animate(self):
        """Plots the found solution."""
        def update(idx):
            global all_lines
            for i, line in enumerate(all_lines):
                if idx >= tours[i].shape[1]:
                    continue
                data = tours[i][:, idx]
                xy_data = line.get_xydata()
                xy_data = np.vstack((xy_data, np.atleast_2d(data)))
                line.set_data(xy_data[:, 0], xy_data[:, 1])
                line.set_linewidth(0.75)
            return all_lines
        path = '/usr/bin/ffmpeg'
        plt.rcParams['animation.ffmpeg_path'] = path
        plt.close('all')
        num_plots = min(int(np.sqrt(len(self.tour_indices))), 3)
        fig, axes = plt.subplots(nrows=num_plots, ncols=num_plots,sharex='col', sharey='row')
        axes = [a for ax in axes for a in ax]
        all_lines = []
        all_tours = []
        for i, ax in enumerate(axes):
            # Convert the indices back into a tour
            idx = self.tour_indices[i]
            if len(idx.size()) == 1:
                idx = idx.unsqueeze(0)
            idx = idx.expand(self.static.size(1), -1)
            data = torch.gather(self.static[i].data, 1, idx).cpu().numpy()
            start = self.static[i, :, 0].cpu().data.numpy()
            x = np.hstack((start[0], data[0], start[0]))
            y = np.hstack((start[1], data[1], start[1]))
            cur_tour = np.vstack((x, y))
            all_tours.append(cur_tour)
            all_lines.append(ax.plot([], [])[0])
            ax.scatter(x, y, s=4, c='r', zorder=2)
            ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)
        tours = all_tours
        anim = FuncAnimation(
            fig, update,
            init_func=None,
            frames=100, interval=200, blit=False,
            repeat=False
        )
        anim.save('line.mp4', dpi=160)
        plt.show()
        exit(0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
print('Detected device {}'.format(dvc))

class Trainer:

    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.simulator = None

    def validate(self, data_loader, save_dir='.', num_plot=5):
        """Used to monitor progress on a validation set & optionally plot solution."""
        self.actor.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        rewards = []
        for batch_idx, batch in enumerate(data_loader):
            static, dynamic, x0 = batch
            static = static.to(dvc)
            dynamic = dynamic.to(dvc)
            x0 = x0.to(dvc) if len(x0) > 0 else None
            with torch.no_grad():
                tour_indices, _ = self.actor.forward(static, dynamic, x0)
            self.simulator = Simulate(static,tour_indices)
            reward = self.simulator.reward().mean().item()
            rewards.append(reward)
            if self.simulator.render is not None and batch_idx < num_plot:
                name = f'batch{batch_idx}_{reward:2.4f}.png'
                path = os.path.join(save_dir, name)
                self.simulator.render(path)
        self.actor.train()
        return np.mean(rewards)

    def train(self, args):
        """Constructs the main actor & critic networks, and performs all training."""
        l = ['task', 'num_nodes', 'train_data', 'valid_data', 'batch_size', 'actor_lr', 'critic_lr', 'max_grad_norm']
        task, num_nodes, train_data, valid_data, batch_size, actor_lr, critic_lr, max_grad_norm = map(lambda x: args[x],l)
        now = f'{str(datetime.datetime.now().time())}'.replace(':','_')
        save_dir = os.path.join(task,f'{num_nodes}',now)
        print('Starting training')
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
        valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)
        best_reward = np.inf
        for epoch in range(20):
            self.actor.train()
            self.critic.train()
            times, losses, rewards, critic_rewards = [], [], [], []
            epoch_start = time.time()
            start = epoch_start
            for batch_idx, batch in enumerate(train_loader):
                static, dynamic, x0 = batch
                static = static.to(dvc)
                dynamic = dynamic.to(dvc)
                x0 = x0.to(dvc) if len(x0) > 0 else None
                # Full forward pass through the dataset
                tour_indices, tour_logp = self.actor(static, dynamic, x0)
                # Sum the log probabilities for each city in the tour
                self.simulator = Simulate(static,tour_indices)
                reward = self.simulator.reward()
                # Query the critic for an estimate of the reward
                critic_est = self.critic(static, dynamic).view(-1)
                advantage = (reward - critic_est)
                actor_loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))
                critic_loss = torch.mean(advantage ** 2)
                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                actor_optim.step()
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                critic_optim.step()
                critic_rewards.append(torch.mean(critic_est.detach()).item())
                rewards.append(torch.mean(reward.detach()).item())
                losses.append(torch.mean(actor_loss.detach()).item())
                if (batch_idx + 1) % 100 == 0:
                    end = time.time()
                    times.append(end - start)
                    start = end
                    mean_loss = np.mean(losses[-100:])
                    mean_reward = np.mean(rewards[-100:])
                    print(f'  Batch {batch_idx}/{len(train_loader)}, reward: {mean_reward:2.3f}, loss: {mean_loss:2.4f}, took: {times[-1]:2.4f}s')
            mean_loss = np.mean(losses)
            mean_reward = np.mean(rewards)
            # Save the weights
            epoch_dir = os.path.join(checkpoint_dir, f'{epoch}')
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)
            save_path = os.path.join(epoch_dir, 'actor.pt')
            torch.save(self.actor.state_dict(), save_path)
            save_path = os.path.join(epoch_dir, 'critic.pt')
            torch.save(self.critic.state_dict(), save_path)
            # Save rendering of validation set tours
            valid_dir = os.path.join(save_dir, f'{epoch}')
            mean_valid = self.validate(valid_loader,valid_dir, num_plot=5)
            # Save best model parameters
            if mean_valid < best_reward:
                best_reward = mean_valid
                save_path = os.path.join(save_dir, 'actor.pt')
                torch.save(self.actor.state_dict(), save_path)
                save_path = os.path.join(save_dir, 'critic.pt')
                torch.save(self.critic.state_dict(), save_path)
            print(f'Mean epoch loss/reward: {mean_loss:2.4f}, {mean_reward:2.4f}, {mean_valid:2.4f}, took: {time.time() - epoch_start:2.4f}s \'({np.mean(times):2.4f}s / 100 batches)\n')

def trainVRP(args):
    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)
    print('Starting VRP training')
    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {3: 10, 10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)
    max_load = LOAD_DICT[args.num_nodes]
    train_data = VehicleRoutingDataset(
        args.train_size,
        args.num_nodes,
        max_load,
        MAX_DEMAND,
        args.seed
    )
    print(f'Train data: {train_data}')
    valid_data = VehicleRoutingDataset(
        args.valid_size,
        args.num_nodes,
        max_load,
        MAX_DEMAND,
        args.seed + 1
    )
    actor = DRL4TSP(
        STATIC_SIZE,
        DYNAMIC_SIZE,
        args.hidden_size,
        train_data.update_dynamic,
        train_data.update_mask,
        args.num_layers,
        args.dropout
    ).to(dvc)
    print(f'Actor: {actor}')
    critic = StateCritic(
        STATIC_SIZE,
        DYNAMIC_SIZE,
        args.hidden_size
    ).to(dvc)
    print(f'Critic: {critic}')
    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, dvc))
        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, dvc))
    TrainObj = Trainer(actor,critic)
    if not args.test:
        TrainObj.train(kwargs)
    test_data = VehicleRoutingDataset(
        args.valid_size,
        args.num_nodes,
        max_load,
        MAX_DEMAND,
        args.seed + 2
    )
    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = TrainObj.validate(test_loader, test_dir, num_plot=5)
    print(f'Average tour length: {out}')

parser = ap(description='Combinatorial Optimization')
parser.add_argument('--seed', default=12345, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--task', default='vrp')
parser.add_argument('--nodes', dest='num_nodes', default=10, type=int)
parser.add_argument('--actor_lr', default=5e-4, type=float)
parser.add_argument('--critic_lr', default=5e-4, type=float)
parser.add_argument('--max_grad_norm', default=2., type=float)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--layers', dest='num_layers', default=1, type=int)
parser.add_argument('--train-size',default=1000000, type=int)
parser.add_argument('--valid-size', default=1000, type=int)
parser.add_argument('-f', '--file', required=False)

args = parser.parse_args()
print('NOTE: SETTING CHECKPOINT: ')
args.checkpoint = os.path.join('vrp', '10', '18_02_14.400918' + os.path.sep)
print(args.checkpoint)
args.test = True
print(args)

trainVRP(args)