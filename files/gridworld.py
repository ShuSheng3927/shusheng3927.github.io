#%%
import numpy as np
import matplotlib.pyplot as plt

# Define the MDP parameters
GRID_SIZE = 5
DISCOUNT_FACTOR = 0.5
REWARD_BASE = -1
REWARD_TOP_RIGHT = 10
ACTIONS = ["up", "down", "left", "right"]

# Action mappings (row, col changes)
action_effects = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Function to plot the grid with rewards
def plot_grid_with_rewards(grid):
    cmap = plt.cm.Greens  # Colormap
    norm = plt.Normalize(vmin=grid.min(), vmax=grid.max())  # Normalize data for colormap
    plt.imshow(grid, cmap=cmap, origin="upper")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "black" if luminance > 0.5 else "white"
            plt.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color)
    plt.colorbar(label="Reward")
    plt.title("Grid with Rewards")
    plt.show()

# Helper function to get valid actions for a state
def get_valid_actions(state):
    row, col = state
    valid_actions = []
    for action, (d_row, d_col) in action_effects.items():
        next_row, next_col = row + d_row, col + d_col
        if 0 <= next_row < GRID_SIZE and 0 <= next_col < GRID_SIZE:
            valid_actions.append(action)
    return valid_actions

# Helper function to get next state and reward
def get_next_state_and_reward(state, action):
    row, col = state
    d_row, d_col = action_effects[action]
    next_row, next_col = row + d_row, col + d_col

    return (next_row, next_col), rewards[next_row, next_col]

# Solve the Bellman equation directly using matrix inverse
def solve_bellman_directly():
    num_states = GRID_SIZE * GRID_SIZE
    transition_matrix = np.zeros((num_states, num_states))
    reward_vector = np.zeros(num_states)
    reward_best = np.zeros(num_states)
    
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state_idx = row * GRID_SIZE + col
            reward_vector[state_idx] = rewards[row, col]

            valid_actions = get_valid_actions((row, col))
            reward_acted = []
            for action in valid_actions:
                (next_row, next_col), reward = get_next_state_and_reward((row, col), action)
                next_idx = next_row * GRID_SIZE + next_col
                transition_matrix[state_idx, next_idx] += 1 / len(valid_actions)
                reward_acted.append(reward)
            reward_best[state_idx] = max(reward_acted)

    identity = np.eye(num_states)
    value_vector = np.linalg.inv(identity - DISCOUNT_FACTOR * transition_matrix) @ reward_best
    return value_vector.reshape((GRID_SIZE, GRID_SIZE))

# Compute the optimal value function using value iteration
def value_iteration(tol=1e-6):
    values = np.zeros((GRID_SIZE, GRID_SIZE))
    delta = float("inf")

    while delta > tol:
        delta = 0
        new_values = np.copy(values)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                state = (row, col)
                q_values = []
                valid_actions = get_valid_actions(state)
                for action in valid_actions:
                    (next_row, next_col), reward = get_next_state_and_reward(state, action)
                    q_values.append(reward + DISCOUNT_FACTOR * values[next_row, next_col])
                new_values[row, col] = max(q_values) if q_values else 0
                delta = max(delta, abs(new_values[row, col] - values[row, col]))
        values = new_values

    return values

# Derive the optimal policy from the value function
def get_optimal_policy(values):
    policy = np.full((GRID_SIZE, GRID_SIZE), "", dtype=object)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state = (row, col)
            action_values = {}

            valid_actions = get_valid_actions(state)
            for action in valid_actions:
                (next_row, next_col), reward = get_next_state_and_reward(state, action)
                action_values[action] = reward + DISCOUNT_FACTOR * values[next_row, next_col]

            if action_values:
                policy[row, col] = max(action_values, key=action_values.get)

    return policy

# Compute the optimal policy using policy iteration
def policy_iteration():
    policy = np.full((GRID_SIZE, GRID_SIZE), ACTIONS[0], dtype=object)
    values = np.zeros((GRID_SIZE, GRID_SIZE))
    is_policy_stable = False

    while not is_policy_stable:
        # Policy evaluation
        while True:
            delta = 0
            new_values = np.copy(values)
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    state = (row, col)
                    action = policy[row, col]
                    (next_row, next_col), reward = get_next_state_and_reward(state, action)
                    new_values[row, col] = reward + DISCOUNT_FACTOR * values[next_row, next_col]
                    delta = max(delta, abs(new_values[row, col] - values[row, col]))
            values = new_values
            if delta < 1e-6:
                break

        # Policy improvement
        is_policy_stable = True
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                state = (row, col)
                action_values = {}
                for action in get_valid_actions(state):
                    (next_row, next_col), reward = get_next_state_and_reward(state, action)
                    action_values[action] = reward + DISCOUNT_FACTOR * values[next_row, next_col]

                best_action = max(action_values, key=action_values.get)
                if policy[row, col] != best_action:
                    is_policy_stable = False
                policy[row, col] = best_action

    return values, policy


#%%
# Initialize the reward grid
np.random.seed(42)  # Set a seed for reproducibility
rewards = np.random.randint(-5, 6, size=(GRID_SIZE, GRID_SIZE))
# rewards = np.full((GRID_SIZE, GRID_SIZE), REWARD_BASE)
# rewards[0, GRID_SIZE - 1] = REWARD_TOP_RIGHT

# Solve using Bellman equation directly
optimal_values_direct = solve_bellman_directly()
print("\nOptimal Value Function (Direct Solution):")
print(optimal_values_direct)

# Solve using value iteration
optimal_values_vi = value_iteration()
print("\nOptimal Value Function (Value / Policy Iteration):")
print(optimal_values_vi)

# Solve using policy iteration
optimal_values_pi, optimal_policy_pi = policy_iteration()
# print("\nOptimal Value Function (Policy Iteration):")
# print(optimal_values_pi)

# Derive the optimal policy from direct solution
optimal_policy = get_optimal_policy(optimal_values_direct)
print("\nOptimal Policy (Direct Solution):")
print(optimal_policy)

# Derive the optimal policy from value iteration
optimal_policy = get_optimal_policy(optimal_values_vi)
print("\nOptimal Policy (Value Iteration):")
print(optimal_policy)

# Derive the optimal policy from policy iteration
print("\nOptimal Policy (Policy Iteration):")
print(optimal_policy_pi)

plot_grid_with_rewards(rewards)

#%%

tol = 1e-4
values = np.zeros((GRID_SIZE, GRID_SIZE))
delta = float("inf")
gap = []

while delta > tol:
    delta = 0
    new_values = np.copy(values)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            state = (row, col)
            q_values = []
            valid_actions = get_valid_actions(state)
            for action in valid_actions:
                (next_row, next_col), reward = get_next_state_and_reward(state, action)
                q_values.append(reward + DISCOUNT_FACTOR * values[next_row, next_col])
            new_values[row, col] = max(q_values) if q_values else 0
            delta = max(delta, abs(new_values[row, col] - values[row, col]))
    values = new_values
    gap.append(np.mean((values - optimal_values_direct)**2))

plt.plot(gap, '-o')
plt.xlabel("Number of Iteartions")
plt.ylabel("L2 Distance From Truth")
plt.title("Value Iteration")
plt.show()