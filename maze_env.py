import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super(MazeEnv, self).__init__()

        #  Maze settings 
        self.grid_size = (5, 5)  # 5x5 grid
        self.cell_size = 100  # For pygame rendering

        #  Define maze components 
        self.start_pos = (0, 0)
        self.goal_pos = (4, 3)

        self.bombs = [(1, 1), (1, 3), (3, 0), (3, 2)]
        self.pits = [(0, 2)]
        self.rewards = [(2, 2), (2, 4), (4, 1)]

        #  Action space
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right

        #  Observation space 
        self.observation_space = spaces.Box(
            low=0, high=max(self.grid_size) - 1, shape=(2,), dtype=np.int32
        )

        #  Initialize state 
        self.state = np.array(self.start_pos, dtype=np.int32)

        #  Pygame 
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.state = np.array(self.start_pos, dtype=np.int32)
        return self.state, {}

    def step(self, action):
        """Apply an action to the environment."""

        x, y = self.state

        #  Movement 
        if action == 0 and x > 0:  # Up
            x -= 1
        elif action == 1 and x < self.grid_size[0] - 1:  # Down
            x += 1
        elif action == 2 and y > 0:  # Left
            y -= 1
        elif action == 3 and y < self.grid_size[1] - 1:  # Right
            y += 1

        self.state = np.array([x, y], dtype=np.int32)

        #  Reward Logic 
        reward = -1  # Default step penalty
        terminated = False

        pos = (x, y)

        if pos in self.bombs:
            reward = -20
            terminated = True
        elif pos in self.pits:
            reward = -100
            terminated = True
        elif pos in self.rewards:
            reward = +10
        elif pos == self.goal_pos:
            reward = +50
            terminated = True

        return self.state, reward, terminated, False, {}

    def render(self):
        """Render the maze using pygame."""

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.grid_size[1] * self.cell_size, self.grid_size[0] * self.cell_size)
            )
            pygame.display.set_caption("Maze Environment")
            self.clock = pygame.time.Clock()

        #  Colors 
        white = (255, 255, 255)
        black = (0, 0, 0)
        red = (200, 50, 50)       # Bomb
        orange = (255, 165, 0)    # Pit
        yellow = (230, 230, 50)   # Reward
        green = (50, 200, 50)     # Goal
        blue = (50, 50, 230)      # Agent

        self.window.fill(white)

        #  Draw grid 
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                rect = pygame.Rect(
                    col * self.cell_size, row * self.cell_size,
                    self.cell_size, self.cell_size
                )
                pygame.draw.rect(self.window, black, rect, width=2)

        #  Helper to draw items 
        def draw_cell(position, color):
            rect = pygame.Rect(
                position[1] * self.cell_size + 10,
                position[0] * self.cell_size + 10,
                self.cell_size - 20,
                self.cell_size - 20,
            )
            pygame.draw.rect(self.window, color, rect)

        # Draw components
        for bomb in self.bombs:
            draw_cell(bomb, red)

        for pit in self.pits:
            draw_cell(pit, orange)

        for reward in self.rewards:
            draw_cell(reward, yellow)

        draw_cell(self.goal_pos, green)  # Goal
        draw_cell(tuple(self.state), blue)  # Agent

        pygame.display.flip()
        self.clock.tick(4)

    def close(self):
        """Close pygame window."""
        if self.window is not None:
            pygame.quit()
            self.window = None


from gymnasium.envs.registration import register

register(
    id="MazeEnv-v0",
    entry_point="maze_env:MazeEnv",
)
