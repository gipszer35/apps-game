from collections import deque
import time
import random
import sys

# Conditional top-level import for headless/Colab environment safety
if "google.colab" not in sys.modules:
    import pygame


class Enemy:
    """Handles enemy behavior, state, and pathfinding tracking."""

    def __init__(self, start_position):
        self.position = start_position

    def _get_available_moves(self):
        """Returns the relative coordinates for movement directions."""
        return [
            (0, -1),  # Up
            (0, 1),  # Down
            (-1, 0),  # Left
            (1, 0),  # Right
        ]

    def move(self, grid, target):
        """BFS-based pathfinding towards the player (target)."""
        queue = deque([(self.position, [])])
        visited = {self.position}

        while queue:
            position, path = queue.popleft()

            if position == target:
                if path:
                    self.position = path[
                        0
                    ]  # Take ONLY the next single step of the shortest path
                    return self.position
                return self.position

            x, y = position

            for dx, dy in self._get_available_moves():
                nxt = (x + dx, y + dy)
                nx, ny = nxt

                if (
                    0 <= ny < len(grid)
                    and 0 <= nx < len(grid[0])
                    and grid[ny][nx] != "#"
                    and nxt not in visited
                ):
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))

        return self.position


class Maze:
    """Handles loading, parsing, and storing the structural layout of the maze."""

    def __init__(self, filename="maze.txt"):
        self.filename = filename
        self.grid = []
        self.dots = set()
        self.player_start = (0, 0)
        self.enemy_start = (0, 0)
        self.load_maze()

    def load_maze(self):
        """Reads the layout from a text file, extracting dynamic starting positions."""
        try:
            with open(self.filename, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            # Fallback layout ideal for quick execution in Google Colab notebooks
            lines = [
                "################",
                "#P...    . E.. #",
                "#   #.#.. #  ###",
                "# .....#. #....#",
                "# ##..... #.#..#",
                "# ..#.... #..###",
                "# ....... #....#",
                "# .......  ....#",
                "#.##.##  ####.##",
                "#   .##  .#  .##",
                "#    ....##   .#",
                "#################",
            ]

        for y, line in enumerate(lines):
            row = []
            for x, cell in enumerate(line):
                if cell == "P":
                    self.player_start = (x, y)
                    row.append(" ")
                elif cell == "E":
                    self.enemy_start = (x, y)
                    row.append(" ")
                elif cell == ".":
                    self.dots.add((x, y))
                    row.append(".")
                else:
                    row.append(cell)
            self.grid.append(row)


class MazeRunnerEnv:
    """The central Reinforcement Learning environment coordinating the simulation."""

    class EnvConfig:
        """Configuration class containing rewards, penalties, and action mappings."""

        def __init__(self):
            # Rewards and penalties
            self.dot_reward = 10
            self.death_penalty = -100
            self.completion_reward = 500
            self.wall_penalty = -1
            self.step_penalty = -1

            # Action mapping dictionary
            self.action_mapping = {
                0: (0, -1),  # Up
                1: (0, 1),  # Down
                2: (-1, 0),  # Left
                3: (1, 0),  # Right
            }

    def __init__(self, maze_file="maze.txt", config=None):
        self.maze_file = maze_file
        self.maze = None
        self.player = None
        self.enemy = None

        self.score = 0
        self.collected = 0
        self.steps = 0
        self.done = False

        self.config = config if config is not None else MazeRunnerEnv.EnvConfig()
        # Pygame variables (initialized lazily when render_pygame is called)
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None

    def reset(self):
        """Resets the environment metrics and reinitializes state objects."""
        self.maze = Maze(self.maze_file)
        self.player = self.maze.player_start
        self.enemy = Enemy(self.maze.enemy_start)

        self.score = 0
        self.collected = 0
        self.steps = 0
        self.done = False

        return self.observe(), {}

    def observe(self):
        """Returns the current state representation (observation) to the agent."""
        return {
            "maze": tuple(
                tuple(1 if cell == "#" else 0 for cell in row) for row in self.maze.grid
            ),
            "dots": tuple(sorted(self.maze.dots)),
            "player": self.player,
            "enemy": self.enemy.position,
        }

    def step(self, action):
        """Executes a single interaction step within the environment."""
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        self.steps += 1
        reward = self.config.step_penalty

        # 1. Player movement via configuration mapping
        dx, dy = self.config.action_mapping[action]
        px, py = self.player
        target_pos = (px + dx, py + dy)
        tx, ty = target_pos

        # Check wall collision safely using row count and column count
        if (
            0 <= ty < len(self.maze.grid)
            and 0 <= tx < len(self.maze.grid[0])
            and self.maze.grid[ty][tx] != "#"
        ):
            self.player = target_pos
        else:
            reward += self.config.wall_penalty

        # 2. Check collision immediately after player move
        if self.player == self.enemy.position:
            self.done = True
            reward += self.config.death_penalty
        else:
            # 3. Handle dot collection if still alive
            if self.player in self.maze.dots:
                self.maze.dots.remove(self.player)
                self.collected += 1
                reward += self.config.dot_reward

            # Check win condition
            if not self.maze.dots:
                self.done = True
                reward += self.config.completion_reward
            else:
                # 4. Enemy movement only if the game is not won or lost yet
                self.enemy.move(self.maze.grid, self.player)

                # Check collision again after enemy move
                if self.player == self.enemy.position:
                    self.done = True
                    reward += self.config.death_penalty

        # Accumulate the final reward into the total score exactly once
        self.score += reward

        return self.observe(), reward, self.done, False, {}

    def render_pygame(self, cell_size=40):
        """Renders the environment using Pygame (for local machine execution)."""
        import pygame  # Lazy import to prevent crashes in headless environments like Colab

        rows = len(self.maze.grid)
        cols = len(self.maze.grid[0])

        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (cols * cell_size, rows * cell_size + 50)
            )
            pygame.display.set_caption("Maze Runner - RL Environment")
            self.font = pygame.font.SysFont("Arial", 20)
            self.clock = pygame.time.Clock()
            self.pygame_initialized = True

        self.screen.fill((10, 12, 25))

        # Draw grid walls
        for y, row in enumerate(self.maze.grid):
            for x, cell in enumerate(row):
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                if cell == "#":
                    pygame.draw.rect(self.screen, (35, 65, 120), rect)
                pygame.draw.rect(self.screen, (20, 25, 45), rect, 1)

        # Draw dots
        for x, y in self.maze.dots:
            pygame.draw.circle(
                self.screen,
                (255, 255, 255),
                (x * cell_size + cell_size // 2, y * cell_size + cell_size // 2),
                3,
            )

        # Draw player
        px, py = self.player
        pygame.draw.circle(
            self.screen,
            (255, 220, 30),
            (px * cell_size + cell_size // 2, py * cell_size + cell_size // 2),
            cell_size // 3,
        )

        # Draw enemy
        ex, ey = self.enemy.position
        pygame.draw.circle(
            self.screen,
            (255, 60, 70),
            (ex * cell_size + cell_size // 2, ey * cell_size + cell_size // 2),
            cell_size // 3,
        )

        # Draw text interface
        score_text = self.font.render(
            f"Score: {self.score} | Dots: {self.collected}", True, (255, 255, 255)
        )
        self.screen.blit(score_text, (10, rows * cell_size + 12))

        if self.done:
            message = (
                "CAUGHT! Press R to restart"
                if self.player == self.enemy.position
                else "COMPLETED! Press R to restart"
            )
            text = self.font.render(message, True, (255, 220, 30))
            self.screen.blit(
                text,
                (cols * cell_size // 2 - text.get_width() // 2, rows * cell_size + 12),
            )

        pygame.display.flip()


class QLearningAgent:
    """A tabular Q-learning agent that learns to navigate the maze."""

    def __init__(
        self,
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
    ):
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}  # Maps state string -> list of 4 Q-values

    def _get_state_key(self, obs):
        """Converts the observation dict into a hashable string key for the Q-table."""
        # To prevent the state space from exploding, we compress the observation
        # into a feature: (player_pos, enemy_pos, hash_of_remaining_dots)
        return f"{obs['player']}_{obs['enemy']}_{hash(obs['dots'])}"

    def get_q_values(self, state_key):
        """Returns the Q-values for a state, initializing them to 0 if new."""
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * self.action_size
        return self.q_table[state_key]

    def act(self, obs):
        """Epsilon-greedy action selection."""
        state_key = self._get_state_key(obs)
        q_values = self.get_q_values(state_key)

        # Exploration (Random step)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        # Exploitation (Best known step)
        max_q = max(q_values)
        # Handle ties randomly to ensure stable learning
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, obs, action, reward, next_obs, done):
        """Updates the Q-value based on the Bellman equation."""
        state_key = self._get_state_key(obs)
        next_state_key = self._get_state_key(next_obs)

        current_q = self.get_q_values(state_key)[action]
        max_next_q = max(self.get_q_values(next_state_key)) if not done else 0.0

        # Temporal Difference (TD) target and update
        td_target = reward + self.gamma * max_next_q
        self.q_table[state_key][action] += self.lr * (td_target - current_q)

    def decay_epsilon(self):
        """Reduces exploration over time as the agent becomes smarter."""
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


class GameRunner:
    """Handles different execution modes for the Maze Runner environment:
    1. MANUAL: Interactive keyboard control.
    2. TRAIN: Training loop using the injected RL agent.
    3. REPLAY: Uses the injected agent to play, records it, then replays it.
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def run_manual(self):
        """Mode 1: Interactive local play using keyboard arrows."""
        self.env.reset()
        self.env.render_pygame()

        actions_map = {
            pygame.K_UP: 0,
            pygame.K_DOWN: 1,
            pygame.K_LEFT: 2,
            pygame.K_RIGHT: 3,
        }

        running = True
        fps = 60
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.env.reset()
                    elif event.key in actions_map and not self.env.done:
                        self.env.step(actions_map[event.key])

            self.env.render_pygame()
            clock.tick(fps)
        pygame.quit()

    def run_train(self, episodes=500):
        """Mode 2: Training session loop for reinforcement learning using the injected agent."""
        print(f"--- Starting Training Session ({episodes} episodes) ---")

        for episode in range(episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # 1. Agent chooses action based on its current policy
                action = self.agent.act(obs)

                # 2. Environment executes the action
                next_obs, reward, done, truncated, info = self.env.step(action)

                # 3. Agent learns from the experience transition
                self.agent.learn(obs, action, reward, next_obs, done)

                obs = next_obs
                total_reward += reward

            # Decay exploration rate after each episode completion
            self.agent.decay_epsilon()

            # Print updates every 50 episodes to keep logs clean
            if (episode + 1) % 50 == 0 or episode == 0:
                # Using hasattr checks to ensure compatibility even if the agent is replaced with a non-Q-learning one
                epsilon_val = getattr(self.agent, "epsilon", 0.0)
                q_table_len = len(getattr(self.agent, "q_table", {}))
                print(
                    f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward} | Epsilon: {epsilon_val:.3f} | States: {q_table_len}"
                )

        print("--- Training Complete ---")

    def run_replay_step_by_step(self):
        """Mode 3: Automated execution using the injected agent, records the trajectory,
        then replays it step-by-step with a 0.5-second delay.
        """
        import time

        print("Generating a game trajectory using the agent...")
        obs, info = self.env.reset()

        # 1. Record phase (playing a game using the agent's logic)
        history = []
        history.append(
            {
                "player": self.env.player,
                "enemy": self.env.enemy.position,
                "dots": set(self.env.maze.dots),
                "score": self.env.score,
                "collected": self.env.collected,
                "done": self.env.done,
            }
        )

        # Ensure we record using the agent's decisions without exploration (optional, or standard act)
        while not self.env.done and len(history) < 200:
            action = self.agent.act(obs)
            next_obs, reward, done, truncated, info = self.env.step(action)
            obs = next_obs

            history.append(
                {
                    "player": self.env.player,
                    "enemy": self.env.enemy.position,
                    "dots": set(self.env.maze.dots),
                    "score": self.env.score,
                    "collected": self.env.collected,
                    "done": self.env.done,
                }
            )

        # 2. Step-by-step playback phase
        print(f"Playing back {len(history)} steps with 0.5s intervals...")
        self.env.reset()
        self.env.render_pygame()

        for step_idx, snapshot in enumerate(history):
            self.env.player = snapshot["player"]
            self.env.enemy.position = snapshot["enemy"]
            self.env.maze.dots = snapshot["dots"]
            self.env.score = snapshot["score"]
            self.env.collected = snapshot["collected"]
            self.env.done = snapshot["done"]

            self.env.render_pygame()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            print(f"Step {step_idx}: Score = {snapshot['score']}")
            time.sleep(0.5)

        pygame.quit()


if __name__ == "__main__":
    env = MazeRunnerEnv()
    agent = QLearningAgent(action_size=4)  # Or any other agent you build later
    runner = GameRunner(env, agent)

    # Choose your mode:
    runner.run_manual()
    runner.run_train(episodes=1000)
    # runner.run_replay_step_by_step()
