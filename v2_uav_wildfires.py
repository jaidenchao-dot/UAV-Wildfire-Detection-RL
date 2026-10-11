# -*- coding: utf-8 -*-
"""V2 UAV WILDFIRES - Multi-Drone Selection & Episode Config"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
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

# --- CUSTOM GRAPHICAL CONFIGURATION SCREEN ---
def select_drone_model():
    plt.ioff()
    fig, ax = plt.subplots(figsize=(9, 6.2))
    plt.subplots_adjust(bottom=0.32)
    ax.set_title("UAV Wildfire Patrol: Configuration Setup", fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')
    
    desc_text = (
        "1. Select a drone model:\n"
        "  • Heavy Duty (Max Steps: 30 | Radius: 10 units)\n"
        "  • All-In-One (Max Steps: 50 | Radius: 5 units)\n"
        "  • Explorer (Max Steps: 100 | Radius: 2 units)\n\n"
        "2. Enter number of training episodes (300 to 6000):"
    )
    ax.text(0.05, 0.65, desc_text, fontsize=10.5, family='sans-serif', verticalalignment='center')
    
    selected_config = {}
    
    # Text box for episodes input
    ax_box = plt.axes([0.28, 0.38, 0.44, 0.07])
    text_box = TextBox(ax_box, 'Episodes: ', initial='300')
    
    # Buttons for drone selection
    ax_btn1 = plt.axes([0.08, 0.12, 0.25, 0.1])
    ax_btn2 = plt.axes([0.37, 0.12, 0.25, 0.1])
    ax_btn3 = plt.axes([0.66, 0.12, 0.25, 0.1])
    
    btn1 = Button(ax_btn1, 'Heavy Duty', color='lightblue', hovercolor='skyblue')
    btn2 = Button(ax_btn2, 'All-In-One', color='lightgreen', hovercolor='limegreen')
    btn3 = Button(ax_btn3, 'Explorer', color='lightyellow', hovercolor='khaki')
    
    def get_episodes():
        try:
            val = int(text_box.text)
            return max(300, min(6000, val))
        except ValueError:
            return 300
            
    def set_heavy(event):
        selected_config.update({
            'name': 'Heavy Duty', 
            'max_steps': 30, 
            'discovery_radius': 10,
            'num_episodes': get_episodes()
        })
        plt.close(fig)
        
    def set_all(event):
        selected_config.update({
            'name': 'All-In-One', 
            'max_steps': 50, 
            'discovery_radius': 5,
            'num_episodes': get_episodes()
        })
        plt.close(fig)
        
    def set_explorer(event):
        selected_config.update({
            'name': 'Explorer', 
            'max_steps': 100, 
            'discovery_radius': 2,
            'num_episodes': get_episodes()
        })
        plt.close(fig)
        
    btn1.on_clicked(set_heavy)
    btn2.on_clicked(set_all)
    btn3.on_clicked(set_explorer)
    
    plt.show(block=True)
    
    if not selected_config:
        selected_config.update({
            'name': 'Explorer', 
            'max_steps': 100, 
            'discovery_radius': 2,
            'num_episodes': 300
        })
        
    print(f"Configuration Selected -> Drone: {selected_config['name']} | Max Steps: {selected_config['max_steps']} | Radius: {selected_config['discovery_radius']} | Episodes: {selected_config['num_episodes']}")
    
    plt.ion()
    return selected_config

# Launch configuration window before initializing environment
drone_spec = select_drone_model()

# set up matplotlib for training
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
    def __init__(self, config):
        super().__init__()
        self.action_space = spaces.Discrete(9)
        
        self.drone_name = config['name']
        self.max_steps = config['max_steps']
        self.discovery_radius = config['discovery_radius']

        # Real-world scaled observation bounds with dynamic max steps
        self.observation_space = spaces.Box(
            low=np.array([-250.0, -100.0, 0, 0.0, 0.0]),
            high=np.array([250.0, 100.0, float(self.max_steps), 100.0, 100.0]),
            dtype=np.float32
        )

        self.grid_x = 250
        self.grid_y = 100

        # Ranger Stations: Main station at index 0, followed by the 6 custom stations[cite: 3]
        self.stations = [
            np.array([135.0, 45.0], dtype=np.float32),  # Main Home Center Station[cite: 3]
            np.array([124.0, 83.0], dtype=np.float32),  # Station 1[cite: 3]
            np.array([182.0, 54.0], dtype=np.float32),  # Station 2[cite: 3]
            np.array([135.0, 18.0], dtype=np.float32),  # Station 3[cite: 3]
            np.array([90.0, 39.0], dtype=np.float32),   # Station 4[cite: 3]
            np.array([42.0, 50.0], dtype=np.float32),   # Station 5[cite: 3]
            np.array([96.0, 6.0], dtype=np.float32)     # Station 6[cite: 3]
        ]

        # Drone starts at the main home center station (index 0)[cite: 3]
        self.start_pos = self.stations[0]

        # Load the shapefile and dynamically scale it to fit the 250x100 RL environment grid[cite: 1, 3]
        print("Loading and scaling shapefile boundary...")
        boundary_file = "samo_tracts/samo_boundary.shp"
        park_boundary_gdf = gpd.read_file(boundary_file)
        raw_polygon = park_boundary_gdf.geometry.unary_union
        
        minx, miny, maxx, maxy = raw_polygon.bounds
        scale_x = self.grid_x / (maxx - minx)
        scale_y = self.grid_y / (maxy - miny)
        
        translated_poly = shapely.affinity.translate(raw_polygon, xoff=-minx, yoff=-miny)
        self.boundary_polygon = shapely.affinity.scale(translated_poly, xfact=scale_x, yfact=scale_y, origin=(0, 0))

        self.valid_mask = np.zeros((self.grid_x, self.grid_y), dtype=bool)
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                if self.boundary_polygon.contains(Point(i, j)):
                    self.valid_mask[i, j] = True

        self.temperature_grid = np.zeros((self.grid_x, self.grid_y), dtype=np.float32)
        self.fire_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.fire_scouted = False

    def _generate_scenario(self):
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

        if self.boundary_polygon.contains(Point(proposed_x, proposed_y)):
            self.pos[0] = proposed_x
            self.pos[1] = proposed_y

        self.steps += 1
        self.trajectory.append(self.pos.copy())

        # Automatically scale battery drain to use 100% capacity over the specified max steps[cite: 3]
        self.battery -= (100.0 / self.max_steps)

        reward = 1.0

        x, y = int(self.pos[0]), int(self.pos[1])
        current_temp = 50.0
        if 0 <= x < self.grid_x and 0 <= y < self.grid_y and self.valid_mask[x, y]:
            val = self.temperature_grid[x, y]
            if not np.isnan(val):
                current_temp = val

        # Thermal reward shaping[cite: 3]
        reward += (current_temp / 20.0)

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
                    reward += 150.0  # Home Station reward[cite: 3]
                else:
                    reward += 250.0  # Station reward[cite: 3]
            else:
                reward -= 100.0  # Stranded penalty[cite: 3]

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

env = AreaDiscoveryEnv(drone_spec)
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)
steps_done = 0
all_trajectories = [] 
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

def plot_trajectories(show_result=False):
    plt.figure(figsize=(10, 4.5))
    plt.clf()

    plt.xlim(0, 250)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal', adjustable='box')

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

    if isinstance(env.boundary_polygon, Polygon):
        x, y = env.boundary_polygon.exterior.xy
        plt.plot(x, y, 'k-', linewidth=2, label="Mountain Boundary", zorder=2)
    elif isinstance(env.boundary_polygon, MultiPolygon):
        for idx, geom in enumerate(env.boundary_polygon.geoms):
            x, y = geom.exterior.xy
            label = "Mountain Boundary" if idx == 0 else ""
            plt.plot(x, y, 'k-', linewidth=2, label=label, zorder=2)

    plt.scatter([env.fire_pos[0]], [env.fire_pos[1]], color='darkred', marker='*', s=250, zorder=6, label="Wildfire Location")

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
    plt.title(f'Santa Monica Mountains Patrol ({drone_spec["name"]} - Radius: {drone_spec["discovery_radius"]})', fontsize=12)

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

# Use the number of episodes specified by the user in the configuration window
num_episodes = drone_spec['num_episodes']

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

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            all_trajectories.append(env.trajectory)
            episode_rewards.append(ep_reward)

            if (i_episode + 1) % 20 == 0:
                plot_trajectories()
            break

print('Training Complete')
plot_trajectories(show_result=True)
plt.ioff()
plt.show()

plt.figure(2)
plt.plot(episode_rewards)
plt.title(f'Reward over Time ({drone_spec["name"]})')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()