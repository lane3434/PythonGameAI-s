import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

# ============================
# Global Constants & Settings
# ============================
BOARD_WIDTH = 20
BOARD_HEIGHT = 20

# RL network parameters
STATE_SIZE = 16   # see get_state() in Snake below
ACTION_SIZE = 3   # actions: 0 = straight, 1 = turn left, 2 = turn right
GAMMA = 0.9
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
TARGET_UPDATE_FREQ = 100  # training steps frequency for updating target network

global_best_score = 0

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# DQN Neural Network and Replay Buffer
# ============================
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ============================
# Snake Agent (DQN trainer per snake)
# ============================
class SnakeAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_counter = 0
        self.target_model.load_state_dict(self.model.state_dict())
        
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target = rewards + GAMMA * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_counter += 1
        if self.update_counter % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
            
    def set_weights(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.target_model.load_state_dict(state_dict)

# ============================
# Snake Game Logic
# ============================
class Snake:
    def __init__(self, agent, color, board_width, board_height):
        self.agent = agent
        self.color = color
        self.board_width = board_width
        self.board_height = board_height
        self.reset()
        
    def reset(self):
        self.body = []
        init_x = random.randint(5, self.board_width - 5)
        init_y = random.randint(5, self.board_height - 5)
        self.head = (init_x, init_y)
        self.body.append(self.head)
        self.direction = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])
        self.alive = True
        self.score = 0
        self.steps = 0
        
    def get_state(self, food_pos):
        head_x, head_y = self.head
        norm_x = head_x / self.board_width
        norm_y = head_y / self.board_height
        
        if self.direction == (0, -1):
            dir_vec = [1, 0, 0, 0]
        elif self.direction == (1, 0):
            dir_vec = [0, 1, 0, 0]
        elif self.direction == (0, 1):
            dir_vec = [0, 0, 1, 0]
        elif self.direction == (-1, 0):
            dir_vec = [0, 0, 0, 1]
        else:
            dir_vec = [0, 0, 0, 0]
        
        left_wall   = head_x / self.board_width
        right_wall  = (self.board_width - head_x) / self.board_width
        top_wall    = head_y / self.board_height
        bottom_wall = (self.board_height - head_y) / self.board_height
        
        food_dx = (food_pos[0] - head_x) / self.board_width
        food_dy = (food_pos[1] - head_y) / self.board_height
        
        length_norm = len(self.body) / (self.board_width * self.board_height)
        
        danger_straight = 0
        danger_left = 0
        danger_right = 0
        if self._is_danger(self._get_next_position(self.direction)):
            danger_straight = 1
        if self._is_danger(self._get_next_position(self._turn_left(self.direction))):
            danger_left = 1
        if self._is_danger(self._get_next_position(self._turn_right(self.direction))):
            danger_right = 1
        
        state = [norm_x, norm_y] + dir_vec + \
                [left_wall, right_wall, top_wall, bottom_wall] + \
                [food_dx, food_dy, length_norm, danger_straight, danger_left, danger_right]
        return np.array(state, dtype=float)
    
    def _get_next_position(self, direction):
        dx, dy = direction
        return (self.head[0] + dx, self.head[1] + dy)
    
    def _turn_left(self, direction):
        dx, dy = direction
        return (-dy, dx)
    
    def _turn_right(self, direction):
        dx, dy = direction
        return (dy, -dx)
    
    def _is_danger(self, pos):
        x, y = pos
        if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
            return True
        if pos in self.body:
            return True
        return False
    
    def move(self, action):
        if action == 1:
            self.direction = self._turn_left(self.direction)
        elif action == 2:
            self.direction = self._turn_right(self.direction)
        new_head = self._get_next_position(self.direction)
        self.head = new_head
        self.body.insert(0, new_head)
        
    def update(self, food_pos):
        self.steps += 1
        reward = 0
        done = False
        
        x, y = self.head
        if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
            self.alive = False
            reward = -10
            done = True
            return reward, done
        
        if self.head in self.body[1:]:
            self.alive = False
            reward = -10
            done = True
            return reward, done
        
        if self.head == food_pos:
            self.score += 10
            reward = 10
        else:
            self.body.pop()
            reward = -0.1
        
        return reward, done

# ----------------------------
# Food (common to all snakes)
# ----------------------------
class Food:
    def __init__(self, board_width, board_height):
        self.board_width = board_width
        self.board_height = board_height
        self.position = self._get_random_position()
        
    def _get_random_position(self):
        return (random.randint(0, self.board_width - 1),
                random.randint(0, self.board_height - 1))
    
    def reposition(self, snakes):
        self.position = self._get_random_position()

# ----------------------------
# Game Environment (handles all snakes and food)
# ----------------------------
class Game:
    def __init__(self, num_snakes, board_width, board_height):
        self.board_width = board_width
        self.board_height = board_height
        self.num_snakes = num_snakes
        self.snakes = []
        self.food = Food(board_width, board_height)
        self.global_best_score = 0
        colors = ["red", "green", "blue", "orange", "purple",
                  "cyan", "magenta", "yellow", "pink", "lime"]
        for i in range(num_snakes):
            agent = SnakeAgent(STATE_SIZE, ACTION_SIZE, device)
            best_model_path = "best_model.pth"
            if os.path.exists(best_model_path):
                agent.load_model(best_model_path)
            color = colors[i % len(colors)]
            snake = Snake(agent, color, board_width, board_height)
            self.snakes.append(snake)
            
    def step(self):
        for snake in self.snakes:
            if not snake.alive:
                continue
            state = snake.get_state(self.food.position)
            action = snake.agent.get_action(state)
            old_distance = np.linalg.norm(np.array(snake.head) - np.array(self.food.position))
            snake.move(action)
            reward, done = snake.update(self.food.position)
            new_distance = np.linalg.norm(np.array(snake.head) - np.array(self.food.position))
            distance_reward = (old_distance - new_distance) * 0.5
            reward += distance_reward
            new_state = snake.get_state(self.food.position)
            snake.agent.remember(state, action, reward, new_state, done)
            snake.agent.train_step()
            
            if snake.head == self.food.position:
                self.food.reposition(self.snakes)
                
            if not snake.alive:
                if snake.score > self.global_best_score:
                    self.global_best_score = snake.score
                    snake.agent.save_model("best_model.pth")
                    best_state_dict = snake.agent.model.state_dict()
                    for s in self.snakes:
                        s.agent.set_weights(best_state_dict)
                snake.reset()

# ============================
# Tkinter GUI (single full-screen window)
# ============================
class SnakeGameGUI:
    def __init__(self, root, game):
        self.root = root
        self.game = game
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.canvas = tk.Canvas(root, width=self.screen_width, height=self.screen_height)
        self.canvas.pack(fill="both", expand=True)
        # view_mode can be "all", "best", or "split"
        self.view_mode = "all"
        self.dark_mode = False
        self.root.bind("<Key>", self.on_key_press)
        self.info_label = tk.Label(root, text="S: Save | M: Toggle All/Best | P: Split view | T: Toggle Dark/Light",
                                    bg="white", fg="black")
        self.info_label.place(relx=0.5, rely=0.95, anchor="center")
        self.update_gui()
        
    def on_key_press(self, event):
        char = event.char.lower()
        if char == 's':
            best_snake = max(self.game.snakes, key=lambda s: s.score)
            best_snake.agent.save_model("best_model.pth")
            print("Model saved to best_model.pth")
        elif char == 'm':
            if self.view_mode == "split":
                self.view_mode = "all"
            else:
                self.view_mode = "best" if self.view_mode == "all" else "all"
            print("View mode:", self.view_mode)
        elif char == 'p':
            self.view_mode = "split"
            print("View mode: split")
        elif char == 't':
            self.dark_mode = not self.dark_mode
            if self.dark_mode:
                self.root.configure(bg="black")
                self.canvas.configure(bg="black")
                self.info_label.configure(bg="black", fg="white")
            else:
                self.root.configure(bg="white")
                self.canvas.configure(bg="white")
                self.info_label.configure(bg="white", fg="black")
            print("Dark mode:", self.dark_mode)
            
    def update_gui(self):
        self.game.step()
        self.draw()
        self.root.after(50, self.update_gui)
        
    def draw(self):
        self.canvas.delete("all")
        if self.view_mode == "split":
            self.draw_split_view()
        else:
            self.draw_single_view()
            
    def draw_single_view(self):
        cell_size = min(self.screen_width / BOARD_WIDTH, self.screen_height / BOARD_HEIGHT)
        offset_x = (self.screen_width - BOARD_WIDTH * cell_size) / 2
        offset_y = (self.screen_height - BOARD_HEIGHT * cell_size) / 2
        snakes_to_draw = self.game.snakes if self.view_mode == "all" else [max(self.game.snakes, key=lambda s: s.score)]
        self.draw_board(offset_x, offset_y, cell_size, snakes_to_draw, self.game.food)
        
    def draw_split_view(self):
        left_width = self.screen_width / 2
        cell_size = min(left_width / BOARD_WIDTH, self.screen_height / BOARD_HEIGHT)
        left_offset_x = (left_width - BOARD_WIDTH * cell_size) / 2
        left_offset_y = (self.screen_height - BOARD_HEIGHT * cell_size) / 2
        right_offset_x = self.screen_width/2 + (left_width - BOARD_WIDTH * cell_size) / 2
        right_offset_y = left_offset_y
        
        # Draw left board (All snakes)
        self.draw_board(left_offset_x, left_offset_y, cell_size, self.game.snakes, self.game.food)
        # Draw right board (Best snake)
        best_snake = max(self.game.snakes, key=lambda s: s.score)
        self.draw_board(right_offset_x, right_offset_y, cell_size, [best_snake], self.game.food)
        
        # Draw vertical separator line
        self.canvas.create_line(self.screen_width/2, 0, self.screen_width/2, self.screen_height,
                                fill="white", dash=(4, 4), width=2)
        
        # Draw text labels at fixed positions on top of each half
        text_fill = "white" if self.dark_mode else "black"
        self.canvas.create_text(self.screen_width/4, 30,
                                text="All Snakes", fill=text_fill,
                                font=("Arial", 16, "bold"))
        self.canvas.create_text(3*self.screen_width/4, 30,
                                text="Best Snake", fill=text_fill,
                                font=("Arial", 16, "bold"))
        
    def draw_board(self, offset_x, offset_y, cell_size, snakes, food):
        # Draw grid lines
        for i in range(BOARD_WIDTH + 1):
            x = offset_x + i * cell_size
            self.canvas.create_line(x, offset_y, x, offset_y + BOARD_HEIGHT * cell_size,
                                    fill="white", dash=(2, 4))
        for j in range(BOARD_HEIGHT + 1):
            y = offset_y + j * cell_size
            self.canvas.create_line(offset_x, y, offset_x + BOARD_WIDTH * cell_size, y,
                                    fill="white", dash=(2, 4))
        
        # Draw food (yellow)
        fx, fy = food.position
        x1 = offset_x + fx * cell_size
        y1 = offset_y + fy * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="yellow")
        
        # Draw snakes
        for snake in snakes:
            for segment in snake.body:
                x, y = segment
                sx1 = offset_x + x * cell_size
                sy1 = offset_y + y * cell_size
                sx2 = sx1 + cell_size
                sy2 = sy1 + cell_size
                self.canvas.create_rectangle(sx1, sy1, sx2, sy2, fill=snake.color)

# ============================
# Start Screen: choose number of snakes
# ============================
class StartScreen:
    def __init__(self, root):
        self.root = root
        self.frame = tk.Frame(root, bg=root["bg"])
        self.frame.place(relx=0.5, rely=0.5, anchor="center")
        self.label = tk.Label(self.frame, text="Enter number of snakes:")
        self.label.pack(side="left")
        self.entry = tk.Entry(self.frame)
        self.entry.pack(side="left")
        self.entry.insert(0, "5")
        self.start_button = tk.Button(self.frame, text="Start Game", command=self.start_game)
        self.start_button.pack(side="left")
        
    def start_game(self):
        try:
            num_snakes = int(self.entry.get())
        except:
            num_snakes = 5
        self.frame.destroy()
        game = Game(num_snakes, BOARD_WIDTH, BOARD_HEIGHT)
        SnakeGameGUI(self.root, game)

# ============================
# Main Entry Point
# ============================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Snake RL Game")
    root.attributes("-fullscreen", True)
    root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))
    root.configure(bg="white")
    
    StartScreen(root)
    root.mainloop()
