# generate_dataset.py
import numpy as np
from case_closed_game import Game, Direction
from tqdm import tqdm
import random

def simple_policy(game, me):
    """A deterministic decent policy for data collection."""
    agent = game.agent1 if me == 1 else game.agent2
    x, y = agent.trail[-1]

    # Try 4 actions from best → worst
    directions = [Direction.RIGHT, Direction.DOWN, Direction.UP, Direction.LEFT]

    for i in range(4):
        d = random.choice(directions)
        directions.remove(d)
        dx, dy = d.value
        new_x = (x + dx) % game.board.width
        new_y = (y + dy) % game.board.height

        # avoid walls (agent trails)
        if game.board.grid[new_y][new_x] == 0:
            return d

    return Direction.RIGHT   # fall back

def board_to_channels(game, me):
    """Encode board as 2 channels: my trail, enemy trail."""
    board = np.zeros((2, game.board.height, game.board.width), dtype=np.float32)

    me_agent = game.agent1 if me == 1 else game.agent2
    enemy = game.agent2 if me == 1 else game.agent1

    for x, y in me_agent.trail:
        board[0, y, x] = 1.0

    for x, y in enemy.trail:
        board[1, y, x] = 1.0

    return board

N_GAMES = 35000
samples = []
labels = []

for _ in tqdm(range(N_GAMES)):
    g = Game()

    while g.agent1.alive and g.agent2.alive and g.turns < 200:
        # states
        s1 = board_to_channels(g, 1)
        s2 = board_to_channels(g, 2)

        # actions
        a1 = simple_policy(g, 1)
        a2 = simple_policy(g, 2)

        samples.append(s1)
        labels.append(list(Direction).index(a1))

        samples.append(s2)
        labels.append(list(Direction).index(a2))

        g.step(a1, a2)

np.savez("data/samples.npz", X=np.array(samples), y=np.array(labels))
print("Saved", len(samples), "samples")

try:
    data = np.load('data/samples.npz')

    print("successful")
except FileNotFoundError:
    print("Error: Make sure the 'data/samples.npz' file exists in the correct location.")
    exit()

print("--- Array Keys ---")
print(f"Keys found in the NPZ file: {list(data.keys())}")

# Access the arrays
X_loaded = data['X']
y_loaded = data['y']

# Verify the shapes
print("\n--- Shape Verification ---")
print(f"X array shape (Samples): {X_loaded.shape}")
print(f"y array shape (Labels): {y_loaded.shape}")