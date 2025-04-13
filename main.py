import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from scipy.spatial.distance import cdist
from matplotlib import font_manager as fm
from IPython.display import HTML
import japanize_matplotlib


# ---------- マップ設定 ----------
map_size = 1000
num_sakura = 60
num_people = 400
num_toilets = 5

# ---------- 座標生成 ----------
sakura_coords = np.array([
    (random.randint(0, map_size-1), random.randint(0, map_size-1))
    for _ in range(num_sakura)
])
people_coords = np.array([
    (random.randint(0, map_size-1), random.randint(0, map_size-1))
    for _ in range(num_people)
])
toilet_coords = np.array([
    (random.randint(0, map_size-1), random.randint(0, map_size-1))
    for _ in range(num_toilets)
])

# ---------- グリッド生成 ----------
grid_x, grid_y = np.meshgrid(np.arange(map_size), np.arange(map_size))
grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

# ---------- 距離計算 ----------
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

dist_to_sakura = np.min(cdist(grid_coords, sakura_coords), axis=1)
dist_to_people = np.min(cdist(grid_coords, people_coords), axis=1)
dist_to_toilets = np.min(cdist(grid_coords, toilet_coords), axis=1)

sakura_score = 1 - normalize(dist_to_sakura)
people_score = normalize(dist_to_people)
toilet_score = 1 - np.abs(normalize(dist_to_toilets) - 0.5) * 2

# ---------- 桜の密集度スコア ----------
def compute_sakura_density(map_size, sakura_coords, radius=5):
    density = np.zeros((map_size, map_size))
    for x, y in sakura_coords:
        x_min = max(0, x - radius)
        x_max = min(map_size, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(map_size, y + radius + 1)
        density[y_min:y_max, x_min:x_max] += 1
    return normalize(density)

sakura_density = compute_sakura_density(map_size, sakura_coords)

# ---------- アニメーション ----------
weight_steps = 20
fig, ax = plt.subplots(figsize=(8, 6))

def animate(i):
    ax.clear()

    # 重みの調整
    w_sakura = 0.3 - 0.01 * i
    w_people = 0.2
    w_toilet = 0.2 
    w_density = 0.3 + 0.01 * i
    score = (
        w_sakura * sakura_score +
        w_people * people_score +
        w_toilet * toilet_score +
        w_density * sakura_density.ravel()
    )
    score_grid = score.reshape((map_size, map_size))
    best_idx = np.argmax(score)
    best_coord = grid_coords[best_idx]

    # 描画
    im = ax.imshow(score_grid, origin='lower')
    ax.scatter(sakura_coords[:, 0], sakura_coords[:, 1], marker='*', label='桜の木', s=60, c='r')
    ax.scatter(people_coords[:, 0], people_coords[:, 1], marker='8', label='他の花見客')
    ax.scatter(toilet_coords[:, 0], toilet_coords[:, 1], marker='s', label='トイレ', c='white')
    ax.scatter(best_coord[0], best_coord[1], marker='X', s=150, label='最適な位置', edgecolors='black')
    ax.set_title(f"桜密集度の重視レベル: {w_density:.2f}")
    ax.set_xlabel("X座標")
    ax.set_ylabel("Y座標")
    ax.legend(loc='upper right')
    return [im]

ani = animation.FuncAnimation(fig, animate, frames=weight_steps, interval=800, blit=False)

# ---------- 表示 ----------
HTML(ani.to_jshtml())
