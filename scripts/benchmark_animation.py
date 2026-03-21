# benchmark_animation.py
# Compares create_cheetah_xy_animation (GIF) vs create_cheetah_xy_video (MP4).

import time
import numpy as np
from max.visualizers import create_cheetah_xy_animation, create_cheetah_xy_video

# Synthetic trajectory: rootx progresses forward, joints vary sinusoidally
T = 501
t = np.linspace(0, 10, T)
states = np.zeros((T, 18))
states[:, 0] = t * 1.2                      # rootx: moves forward
states[:, 1] = 0.05 * np.sin(2 * np.pi * t) # rootz: slight bounce
states[:, 2] = 0.05 * np.sin(t)             # rooty: slight pitch
states[:, 3] = 0.8 * np.sin(2 * t)          # bthigh
states[:, 4] = 0.5 * np.sin(2 * t + 0.5)   # bshin
states[:, 5] = 0.3 * np.sin(2 * t + 1.0)   # bfoot
states[:, 6] = 0.8 * np.sin(2 * t + np.pi) # fthigh
states[:, 7] = 0.5 * np.sin(2 * t + np.pi + 0.5)
states[:, 8] = 0.3 * np.sin(2 * t + np.pi + 1.0)
states[:, 9] = 1.2 + 0.1 * np.sin(t)       # forward velocity

print(f"Trajectory: {T} steps  →  subsampled to 300 frames\n")

print("Running GIF (create_cheetah_xy_animation)...")
t0 = time.perf_counter()
gif_path = create_cheetah_xy_animation(states)
gif_time = time.perf_counter() - t0
print(f"  GIF done: {gif_time:.2f}s  →  {gif_path}")

print("\nRunning MP4 (create_cheetah_xy_video)...")
t0 = time.perf_counter()
mp4_path = create_cheetah_xy_video(states)
mp4_time = time.perf_counter() - t0
print(f"  MP4 done: {mp4_time:.2f}s  →  {mp4_path}")

import os
gif_mb = os.path.getsize(gif_path) / 1e6
mp4_mb = os.path.getsize(mp4_path) / 1e6

print(f"\n{'':=<40}")
print(f"  GIF:  {gif_time:.2f}s   {gif_mb:.1f} MB")
print(f"  MP4:  {mp4_time:.2f}s   {mp4_mb:.1f} MB")
print(f"  Speedup: {gif_time / mp4_time:.1f}x faster")
print(f"  Size:    {gif_mb / mp4_mb:.1f}x smaller")
print(f"{'':=<40}")
