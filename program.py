import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from collections import deque
import random
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Display settings
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D Drone Simulation with DQN")

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_DRONE = (0, 128, 255)

# Drone properties
NUMBER_OF_DRONES = 5
DRONE_SIZE = 10
DRONE_VELOCITY = 5
DIRECTION_ACTIONS = [(0, -DRONE_VELOCITY), (0, DRONE_VELOCITY), (-DRONE_VELOCITY, 0), (DRONE_VELOCITY, 0),
                     (-DRONE_VELOCITY, -DRONE_VELOCITY), (-DRONE_VELOCITY, DRONE_VELOCITY),
                     (DRONE_VELOCITY, -DRONE_VELOCITY), (DRONE_VELOCITY, DRONE_VELOCITY), (0, 0)]
NUM_ACTIONS = len(DIRECTION_ACTIONS)

# Deep Reinforcement Learning Q-Network parameters
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1.0
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Replay buffer
MAX_MEMORY_SIZE = 1000
experience_memory = deque(maxlen=MAX_MEMORY_SIZE)

# Metrics
energy_consumption = [0] * NUMBER_OF_DRONES
delay = [0] * NUMBER_OF_DRONES
throughput = [0] * NUMBER_OF_DRONES
collision_count = [0] * NUMBER_OF_DRONES


# Define the DQN model
class DroneDQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DroneDQN, self).__init__()
        self.hidden_layer1 = nn.Linear(input_size, 24)
        self.hidden_layer2 = nn.Linear(24, 24)
        self.output_layer = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        return self.output_layer(x)


# Initialize drones with models and optimizers
STATE_DIMENSIONS = 2  # Drone's x and y positions
drone_agents = []
for idx in range(NUMBER_OF_DRONES):
    pos_x = random.randint(DRONE_SIZE, SCREEN_WIDTH - DRONE_SIZE)
    pos_y = random.randint(DRONE_SIZE, SCREEN_HEIGHT - DRONE_SIZE)
    dqn_model = DroneDQN(STATE_DIMENSIONS, NUM_ACTIONS)
    model_optimizer = optim.Adam(dqn_model.parameters(), lr=LEARNING_RATE)
    drone_agents.append({'position': [pos_x, pos_y], 'model': dqn_model, 'optimizer': model_optimizer})


# Helper functions
def select_action(model, current_state, exploration_rate):
    """Epsilon-greedy action selection"""
    if random.random() < exploration_rate:
        return random.choice(range(NUM_ACTIONS))
    with torch.no_grad():
        q_values = model(torch.FloatTensor(current_state))
        return torch.argmax(q_values).item()


def save_experience(state, action, reward, new_state, completed):
    """Store experience in replay memory"""
    experience_memory.append((state, action, reward, new_state, completed))


def train_dqn(drone_agent):
    """Train the drone's model using experiences in replay memory"""
    if len(experience_memory) < BATCH_SIZE:
        return
    mini_batch = random.sample(experience_memory, BATCH_SIZE)
    for state, action, reward, next_state, done in mini_batch:
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        target_value = reward + (1 - done) * DISCOUNT_FACTOR * torch.max(drone_agent['model'](next_state))
        target_output = drone_agent['model'](state)
        target_output[action] = target_value

        # Perform gradient descent
        drone_agent['optimizer'].zero_grad()
        loss = nn.functional.mse_loss(drone_agent['model'](state)[action], target_value)
        loss.backward()
        drone_agent['optimizer'].step()


def compute_reward(drone_index, future_position):
    """Reward function for moving without collision"""
    reward_value = 1  # Base reward for moving
    next_x, next_y = future_position

    # Calculate distance traveled for energy consumption
    current_pos = drone_agents[drone_index]['position']
    distance_traveled = np.linalg.norm(np.array(current_pos) - np.array(future_position))
    energy_consumption[drone_index] += distance_traveled

    # Increment delay
    delay[drone_index] += 1

    # Penalty if the drone is too close to other drones
    for other_index, other_agent in enumerate(drone_agents):
        if other_index != drone_index:
            distance = np.linalg.norm(np.array(other_agent['position']) - np.array(future_position))
            if distance < 2 * DRONE_SIZE:
                reward_value -= 100  # Penalty for collision
                collision_count[drone_index] += 1

    # Penalty for hitting walls
    if next_x <= DRONE_SIZE or next_x >= SCREEN_WIDTH - DRONE_SIZE or next_y <= DRONE_SIZE or next_y >= SCREEN_HEIGHT - DRONE_SIZE:
        reward_value -= 100
        collision_count[drone_index] += 1

    # Update throughput if no collision
    if reward_value > 0:
        throughput[drone_index] += 1

    return reward_value


# Main loop
frame_clock = pygame.time.Clock()
is_running = True
while is_running:
    window.fill(COLOR_WHITE)  # Clear screen

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False

    for index, drone_agent in enumerate(drone_agents):
        current_position = drone_agent['position']
        action_index = select_action(drone_agent['model'], current_position, EXPLORATION_RATE)
        delta_x, delta_y = DIRECTION_ACTIONS[action_index]
        new_position = [current_position[0] + delta_x, current_position[1] + delta_y]
        reward_value = compute_reward(index, new_position)

        # Update drone's position if within bounds
        next_position = current_position
        if DRONE_SIZE <= new_position[0] <= SCREEN_WIDTH - DRONE_SIZE and DRONE_SIZE <= new_position[
            1] <= SCREEN_HEIGHT - DRONE_SIZE:
            drone_agent['position'] = new_position
            next_position = new_position

        episode_done = reward_value < -99  # Consider done if drone collides
        save_experience(current_position, action_index, reward_value, next_position, episode_done)
        train_dqn(drone_agent)  # Train the model using experiences

        # Draw the drone
        pygame.draw.circle(window, COLOR_DRONE, (int(drone_agent['position'][0]), int(drone_agent['position'][1])),
                           DRONE_SIZE)

    # Update display
    pygame.display.flip()
    frame_clock.tick(30)  # Limit FPS to 30

    # Decay exploration rate
    if EXPLORATION_RATE > MIN_EXPLORATION_RATE:
        EXPLORATION_RATE *= EXPLORATION_DECAY

# Close Pygame
pygame.quit()

# Plot metrics
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(energy_consumption, label="Energy Consumption")
axs[0, 0].set_title("Energy Consumption per Drone")
axs[0, 1].plot(delay, label="Delay", color="orange")
axs[0, 1].set_title("Delay per Drone")
axs[1, 0].plot(throughput, label="Throughput", color="green")
axs[1, 0].set_title("Throughput per Drone")
axs[1, 1].plot(collision_count, label="Collisions", color="red")
axs[1, 1].set_title("Collision Count per Drone")

for ax in axs.flat:
    ax.set(xlabel='Drone Index', ylabel='Value')
    ax.label_outer()
    ax.legend()

plt.tight_layout()
plt.show()
print ("End!")
