# -*- coding: utf-8 -*-
"""Test runner for trained UAV wildfire patrol Q-weights.

Loads weights saved by v2_uav_wildfires.py and flies the drone greedily
(no exploration) through several randomly generated fire scenarios.

Usage (run from the folder that contains samo_tracts/ and weights/):
    python test_q_weights.py                         # best All-In-One weights, 5 cases
    python test_q_weights.py -d "Heavy Duty" -n 10
    python test_q_weights.py -w weights/final_q_weights_Explorer.pt --seed 7
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: figures are saved to files
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon, MultiPolygon

# Import the environment and network from the training script (training does
# not start on import because it is guarded by `if __name__ == "__main__"`).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_uav_wildfires import AreaDiscoveryEnv, DQN, DRONE_SPECS, device, weights_filename, OBS_VERSION


def parse_args():
    p = argparse.ArgumentParser(description="Test trained UAV wildfire patrol Q-weights")
    p.add_argument("-d", "--drone", choices=list(DRONE_SPECS), default="All-In-One",
                   help="Drone model whose best weights to load (ignored if --weights is given)")
    p.add_argument("-w", "--weights", type=str, default=None,
                   help="Path to a specific .pt checkpoint")
    p.add_argument("--weights_folder", type=str, default="weights")
    p.add_argument("-n", "--cases", type=int, default=5, help="Number of test scenarios")
    p.add_argument("--seed", type=int, default=0, help="Base seed; case i uses seed+i")
    p.add_argument("-o", "--output_folder", type=str, default="test_results")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_policy(path):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    net = DQN(ckpt['n_observations'], ckpt['n_actions']).to(device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net, ckpt


def run_case(env, net, seed):
    seed_everything(seed)
    obs, _ = env.reset(seed=seed)
    total_reward, scouted_step = 0.0, None

    while True:
        with torch.no_grad():
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = net(state).argmax(dim=1).item()  # greedy, no epsilon
        was_scouted = env.fire_scouted
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if env.fire_scouted and not was_scouted:
            scouted_step = env.total_steps
        if terminated or truncated:
            break

    dists = [np.linalg.norm(env.pos - s) for s in env.stations]
    nearest = int(np.argmin(dists))
    landed = env.landed
    return {
        'seed': seed,
        'fire_found': env.fire_scouted,
        'found_at_step': scouted_step,
        'steps': env.total_steps,
        'recharges': env.recharges_used,
        'battery': env.battery,
        'landed': landed,
        'station': ("Home" if nearest == 0 else f"Station {nearest}") if landed else "Stranded",
        'reward': total_reward,
        'trajectory': np.array(env.trajectory),
        'recharge_points': np.array(env.recharge_points),
        'fire_pos': env.fire_pos.copy(),
        'temperature_grid': env.temperature_grid.copy(),
    }


def plot_case(env, result, case_num, drone_name, out_path):
    plt.figure(figsize=(10, 4.5))
    plt.xlim(0, 250)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.imshow(result['temperature_grid'].T, origin='lower', extent=[0, env.grid_x, 0, env.grid_y],
               cmap='YlOrRd', alpha=0.6, vmin=0, vmax=100, zorder=1)

    geoms = [env.boundary_polygon] if isinstance(env.boundary_polygon, Polygon) else list(env.boundary_polygon.geoms)
    for idx, geom in enumerate(geoms):
        x, y = geom.exterior.xy
        plt.plot(x, y, 'k-', linewidth=2, label="Mountain Boundary" if idx == 0 else "", zorder=2)

    # Discovery radius around the fire: the drone "finds" it once inside this circle
    plt.gca().add_patch(plt.Circle(result['fire_pos'], env.discovery_radius, color='darkred',
                                   fill=False, linestyle='--', zorder=5))
    plt.scatter(*result['fire_pos'], color='darkred', marker='*', s=250, zorder=6, label="Wildfire Location")

    for idx, s in enumerate(env.stations):
        plt.scatter(*s, color='red' if idx == 0 else 'orange', s=100, zorder=5,
                    label="Main Home Station" if idx == 0 else ("Ranger Stations" if idx == 1 else ""))

    traj = result['trajectory']
    plt.plot(traj[:, 0], traj[:, 1], color='blue', linewidth=2, label="Test Flight Path", zorder=4)
    plt.scatter(*traj[-1], color='blue', marker='X', s=120, zorder=7, label="End Position")

    rp = result['recharge_points']
    if len(rp):
        plt.scatter(rp[:, 0], rp[:, 1], color='limegreen', edgecolors='black', marker='P', s=120,
                    zorder=8, label="Recharge")

    status = "FIRE FOUND" if result['fire_found'] else "fire missed"
    plt.title(f"Test Case {case_num} ({drone_name}) — {status}, {result['station']}, "
              f"{result['recharges']} recharge(s), reward {result['reward']:.0f}", fontsize=11)
    plt.xlabel("East-West Distance Units (~0.16 miles / unit)", fontsize=10)
    plt.ylabel("North-South Distance Units (~0.16 miles / unit)", fontsize=10)
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    weights_path = Path(args.weights) if args.weights else Path(args.weights_folder) / weights_filename(args.drone, "best")
    if not weights_path.exists():
        sys.exit(f"Weights not found: {weights_path}\nRun v2_uav_wildfires.py first "
                 f"(e.g. python v2_uav_wildfires.py -d \"{args.drone}\").")

    net, ckpt = load_policy(weights_path)
    spec = ckpt['drone_spec']  # use the spec the weights were trained with
    print(f"Loaded {weights_path} | Drone: {spec['name']} | trained to episode {ckpt['episode']} | "
          f"{ckpt['window']}-ep avg reward {ckpt['avg_reward']:.1f}")

    out_dir = Path(args.output_folder)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    env = AreaDiscoveryEnv(spec)

    # Weights trained with an older observation layout cannot drive the current environment
    env_obs = env.observation_space.shape[0]
    if ckpt.get('obs_version', 1) != OBS_VERSION or ckpt['n_observations'] != env_obs:
        sys.exit(f"{weights_path} was trained with an older version of the environment "
                 f"({ckpt['n_observations']} inputs; the current one uses {env_obs}).\n"
                 f"Retrain it: python v2_uav_wildfires.py -d \"{spec['name']}\" -e 3000")

    results = []
    for i in range(args.cases):
        r = run_case(env, net, args.seed + i)
        results.append(r)
        plot_case(env, r, i + 1, spec['name'], out_dir / f"test_case_{i + 1}.png")

    # --- Summary table ---
    header = (f"{'Case':>4} {'Seed':>5} {'Fire':>6} {'@Step':>6} {'Steps':>6} {'Recharges':>10} "
              f"{'Battery':>8} {'End':>11} {'Reward':>10}")
    print("\n" + header + "\n" + "-" * len(header))
    for i, r in enumerate(results, 1):
        print(f"{i:>4} {r['seed']:>5} {('yes' if r['fire_found'] else 'no'):>6} "
              f"{(r['found_at_step'] if r['found_at_step'] is not None else '-'):>6} "
              f"{r['steps']:>6} {r['recharges']:>10} {r['battery']:>7.0f}% {r['station']:>11} {r['reward']:>10.1f}")

    n = len(results)
    found = sum(r['fire_found'] for r in results)
    landed = sum(r['landed'] for r in results)
    both = sum(r['fire_found'] and r['landed'] for r in results)
    print("-" * len(header))
    print(f"Fire found: {found}/{n} ({100 * found / n:.0f}%) | Ended with a landing: {landed}/{n} "
          f"({100 * landed / n:.0f}%) | Both: {both}/{n} | "
          f"Avg recharges: {np.mean([r['recharges'] for r in results]):.1f} | "
          f"Mean reward: {np.mean([r['reward'] for r in results]):.1f}")
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()