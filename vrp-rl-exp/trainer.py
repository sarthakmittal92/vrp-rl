import os
import time
import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
from multiprocessing import Pool
from itertools import repeat

from model.base import *
from model.actor import *
from model.critic import *
from vrp import *

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
    if not args.test:
        TrainObj = Trainer(actor,critic)
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
