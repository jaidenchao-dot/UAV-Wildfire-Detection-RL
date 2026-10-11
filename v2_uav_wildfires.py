# -*- coding: utf-8 -*-
"""V2 UAV WILDFIRES - Shapefile Integrated with Thermal Reward Shaping"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Geospatial imports
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import shapely.affinity

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class AreaDiscoveryEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(9)

        # Real-world scaled observation bounds:
        # X: -250 to 250, Y: -100 to 100, steps, battery, temperature (0-100)
        self.observation_space = spaces.Box(
            low=np.array([-250.0, -100.0, 0, 0.0, 0.0]),
            high=np.array([250.0, 100.0, 35, 100.0, 100.0]),
            dtype=np.float32
        )

        self.max_steps = 35
        self.grid_x = 250
        self.grid_y = 100

        # Ranger Stations: Main station at index 0, followed by the 6 custom stations
        self.stations = [
            np.array([135.0, 45.0], dtype=np.float32),  # Main Home Center Station
            np.array([124.0, 83.0], dtype=np.float32),  # Station 1
            np.array([182.0, 54.0], dtype=np.float32),  # Station 2
            np.array([135.0, 18.0], dtype=np.float32),  # Station 3
            np.array([90.0, 39.0], dtype=np.float32),   # Station 4
            np.array([42.0, 50.0], dtype=np.float32),   # Station 5
            np.array([96.0, 6.0], dtype=np.float32)     # Station 6
        ]

        # Drone starts at the main home center station (index 0)
        self.start_pos = self.stations[0]

        # Discovery radius scaled to real life (~0.32 miles / ~1,700 ft per step scan width)
        self.discovery_radius = 2

        # Load the shapefile and dynamically scale it to fit the 250x100 RL environment grid
        print("Loading and scaling shapefile boundary...")
        boundary_file = "samo_tracts/samo_boundary.shp"
        park_boundary_gdf = gpd.read_file(boundary_file)
        raw_polygon = park_boundary_gdf.geometry.unary_union
        
        # Determine the scale required to map the coordinates to 250x100
        minx, miny, maxx, maxy = raw_polygon.bounds
        scale_x = self.grid_x / (maxx - minx)
        scale_y = self.grid_y / (maxy - miny)
        
        # Translate to origin (0,0) and scale up to grid dimensions
        translated_poly = shapely.affinity.translate(raw_polygon, xoff=-minx, yoff=-miny)
        self.boundary_polygon = shapely.affinity.scale(translated_poly, xfact=scale_x, yfact=scale_y, origin=(0, 0))

        # Full map mask spanning 250x100 generated from the loaded polygon
        self.valid_mask = np.zeros((self.grid_x, self.grid_y), dtype=bool)
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                if self.boundary_polygon.contains(Point(i, j)):
                    self.valid_mask[i, j] = True

        self.temperature_grid = np.zeros((self.grid_x, self.grid_y), dtype=np.float32)
        self.fire_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.fire_scouted = False

    def _generate_scenario(self):
        # Generate smooth random temperature map spanning across the full 250x100 space
        self.temperature_grid = np.zeros((self.grid_x, self.grid_y), dtype=np.float32)
        num_hotspots = random.randint(5, 9)
        valid_indices = np.argwhere(self.valid_mask)

        if len(valid_indices) > 0:
            hotspots = valid_indices[np.random.choice(len(valid_indices), num_hotspots, replace=False)]
            x_indices, y_indices = np.indices((self.grid_x, self.grid_y))

            for hx, hy in hotspots:
                dist_sq = (x_indices - hx)**2 + (y_indices - hy)**2
                self.temperature_grid += np.exp(-dist_sq / 1200.0)

            min_v, max_v = self.temperature_grid.min(), self.temperature_grid.max()
            if max_v > min_v:
                self.temperature_grid = (self.temperature_grid - min_v) / (max_v - min_v) * 100.0
            else:
                self.temperature_grid.fill(50.0)

        self.temperature_grid[~self.valid_mask] = np.nan

        # Generate fire location near a super hot location (temperature >= 75.0)
        hot_mask = (self.temperature_grid >= 75.0) & self.valid_mask
        hot_indices = np.argwhere(hot_mask)

        if len(hot_indices) == 0:
            hot_indices = valid_indices

        fire_idx = hot_indices[random.randint(0, len(hot_indices) - 1)]
        self.fire_pos = np.array([float(fire_idx[0]), float(fire_idx[1])], dtype=np.float32)
        self.fire_scouted = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start_pos.copy()
        self.steps = 0
        self.battery = 100.0

        # Generate new temperature layout and fire location for this run
        self._generate_scenario()

        self.visited = ~self.valid_mask.copy()
        self.trajectory = [self.pos.copy()]
        self._mark_visited()

        return self._get_obs(), {}

    def _get_obs(self):
        closest_station = min(self.stations, key=lambda s: np.linalg.norm(self.pos - s))
        dist_x = closest_station[0] - self.pos[0]
        dist_y = closest_station[1] - self.pos[1]
        steps_left = self.max_steps - self.steps

        # Read thermal sensor at the drone's current grid position
        x, y = int(self.pos[0]), int(self.pos[1])
        if 0 <= x < self.grid_x and 0 <= y < self.grid_y and self.valid_mask[x, y]:
            current_temp = self.temperature_grid[x, y]
            if np.isnan(current_temp):
                current_temp = 50.0
        else:
            current_temp = 50.0

        return np.array([dist_x, dist_y, steps_left, self.battery, current_temp], dtype=np.float32)

    def _mark_visited(self):
        x, y = int(self.pos[0]), int(self.pos[1])
        r = self.discovery_radius
        new_tiles = 0

        for i in range(max(0, x-r), min(self.grid_x, x+r+1)):
            for j in range(max(0, y-r), min(self.grid_y, y+r+1)):
                if not self.visited[i, j]:
                    self.visited[i, j] = True
                    new_tiles += 1
        return new_tiles

    def step(self, action):
        dx = (action % 3 - 1) * 2.1
        dy = (action // 3 - 1) * 2.1

        proposed_x = self.pos[0] + dx
        proposed_y = self.pos[1] + dy

        # Ensure bounds mapping remains inside the actual shapefile polygon rules
        if self.boundary_polygon.contains(Point(proposed_x, proposed_y)):
            self.pos[0] = proposed_x
            self.pos[1] = proposed_y

        self.steps += 1
        self.trajectory.append(self.pos.copy())

        self.battery -= (100.0 / self.max_steps)

        reward = 1.0

        # Read thermal sensor at current position for reward shaping
        x, y = int(self.pos[0]), int(self.pos[1])
        current_temp = 50.0
        if 0 <= x < self.grid_x and 0 <= y < self.grid_y and self.valid_mask[x, y]:
            val = self.temperature_grid[x, y]
            if not np.isnan(val):
                current_temp = val

        # THERMAL REWARD SHAPING: Guide the agent toward heat hotspots where fires occur
        reward += (current_temp / 20.0)

        # Check if drone scouted the wildfire within discovery range (+10,000 score booster)
        if not self.fire_scouted:
            dist_to_fire = np.linalg.norm(self.pos - self.fire_pos)
            if dist_to_fire <= self.discovery_radius:
                reward += 10000.0
                self.fire_scouted = True

        distances = [np.linalg.norm(self.pos - station) for station in self.stations]
        min_dist_to_station = min(distances)

        if self.battery <= 20.0 and min_dist_to_station > 9.0:
            reward -= 5.0

        terminated = False
        truncated = False

        if self.steps >= self.max_steps or self.battery <= 0:
            terminated = True
            if min_dist_to_station <= 9.0:
                closest_station_idx = np.argmin(distances)
                if closest_station_idx == 0:
                    reward += 150.0  # Home Station reward
                else:
                    reward += 250.0  # Increased reward for reaching other stations
            else:
                reward -= 100.0  # Stranded penalty

        return self._get_obs(), reward, terminated, truncated, {}

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.005
LR = 1e-3

env = AreaDiscoveryEnv()
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0
all_trajectories = [] # To store paths for visualization
episode_rewards = []

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

import matplotlib.patches as patches

def plot_trajectories(show_result=False):
    plt.figure(figsize=(10, 4.5))
    plt.clf()

    plt.xlim(0, 250)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot Temperature Grid Heatmap across the full 250x100 grid bounds
    plt.imshow(
        env.temperature_grid.T,
        origin='lower',
        extent=[0, env.grid_x, 0, env.grid_y],
        cmap='YlOrRd',
        alpha=0.6,
        vmin=0,
        vmax=100,
        zorder=1
    )

    # Plot Mountain Boundary natively rendering shapefile bounds
    if isinstance(env.boundary_polygon, Polygon):
        x, y = env.boundary_polygon.exterior.xy
        plt.plot(x, y, 'k-', linewidth=2, label="Mountain Boundary", zorder=2)
    elif isinstance(env.boundary_polygon, MultiPolygon):
        for idx, geom in enumerate(env.boundary_polygon.geoms):
            x, y = geom.exterior.xy
            label = "Mountain Boundary" if idx == 0 else ""
            plt.plot(x, y, 'k-', linewidth=2, label=label, zorder=2)

    # Plot Wildfire Location Marker
    plt.scatter([env.fire_pos[0]], [env.fire_pos[1]], color='darkred', marker='*', s=250, zorder=6, label="Wildfire Location")

    # Plot ALL Ranger Stations on the map distinctly
    for idx, station in enumerate(env.stations):
        if idx == 0:
            plt.scatter([station[0]], [station[1]], color='red', s=100, zorder=5, label="Main Home Station")
        else:
            plt.scatter([station[0]], [station[1]], color='orange', s=100, zorder=5, label="Ranger Stations" if idx == 1 else "")

    for traj in all_trajectories[:-1]:
        traj_arr = np.array(traj)
        plt.plot(traj_arr[:, 0], traj_arr[:, 1], color='gray', alpha=0.3, linewidth=1, zorder=3)

    if len(all_trajectories) > 0:
        latest_traj = np.array(all_trajectories[-1])
        plt.plot(latest_traj[:, 0], latest_traj[:, 1], color='blue', linewidth=2, label="Latest Flight Path", zorder=4)

    plt.xlabel("East-West Distance Units (~0.16 miles / unit)", fontsize=10)
    plt.ylabel("North-South Distance Units (~0.16 miles / unit)", fontsize=10)
    plt.title('Santa Monica Mountains Wildfire Patrol & Thermal Scans', fontsize=12)

    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.3)

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episodes = 6000 if torch.cuda.is_available() else 300

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    ep_reward = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        ep_reward += reward
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward_tensor)
        state = next_state

        optimize_model()

        # Soft update target network
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            all_trajectories.append(env.trajectory)
            episode_rewards.append(ep_reward)

            # Update visualization every 20 episodes to save rendering time
            if (i_episode + 1) % 20 == 0:
                plot_trajectories()
            break

print('Training Complete')
plot_trajectories(show_result=True)
plt.ioff()
plt.show()

# Optional: Plot the learning curve
plt.figure(2)
plt.plot(episode_rewards)
plt.title('Reward over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()