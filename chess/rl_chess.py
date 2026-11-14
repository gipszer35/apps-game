import json
import os
import random
import ast
from abc import ABC, abstractmethod

import chess
import time
import copy
import pygame


# ABSTRACT AGENT INTERFACE
class BaseAgent(ABC):

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def decay_epsilon(self):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass


# TABULAR Q-LEARNING AGENT
class QLearningAgent(BaseAgent):

    def __init__(
        self,
        action_size=4,
        lr=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
    ):

        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = {}

    def _get_state_key(self, state):
        return tuple(state)

    def act(self, state):

        s_key = self._get_state_key(state)

        if s_key not in self.q_table:
            self.q_table[s_key] = [0.0] * self.action_size

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        q_vals = self.q_table[s_key]

        max_q = max(q_vals)

        best_actions = [i for i, q in enumerate(q_vals) if q == max_q]

        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):

        s_key = self._get_state_key(state)
        ns_key = self._get_state_key(next_state)

        if s_key not in self.q_table:
            self.q_table[s_key] = [0.0] * self.action_size

        if ns_key not in self.q_table:
            self.q_table[ns_key] = [0.0] * self.action_size

        current_q = self.q_table[s_key][action]

        max_future_q = 0.0 if done else max(self.q_table[ns_key])

        target = reward + self.gamma * max_future_q

        self.q_table[s_key][action] += self.lr * (target - current_q)

    def decay_epsilon(self):

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filename="chess_q_table.json"):

        serializable = {json.dumps(list(k)): v for k, v in self.q_table.items()}

        with open(filename, "w") as f:

            json.dump({"epsilon": self.epsilon, "q_table": serializable}, f, indent=4)

        print(f"Saved Q-table: {len(self.q_table)} states")

    def load(self, filename="chess_q_table.json"):

        if not os.path.exists(filename):
            print("No existing Q-table found.")
            return False

        with open(filename, "r") as f:
            data = json.load(f)

        self.epsilon = data.get("epsilon", self.min_epsilon)

        self.q_table = {
            tuple(json.loads(k)): v for k, v in data.get("q_table", {}).items()
        }

        print(f"Loaded Q-table: {len(self.q_table)} states")

        return True


# CHESS TACTICAL ENVIRONMENT
class ChessTacticalEnv:

    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    ACTION_SIZE = 4

    def __init__(self, max_moves=150):

        self.board = chess.Board()
        self.max_moves = max_moves
        self.move_count = 0

    def reset(self):

        self.board.reset()
        self.move_count = 0

        return self.get_state()

    def _get_material_score(self, color):

        score = 0

        for piece_type, value in self.PIECE_VALUES.items():

            score += len(self.board.pieces(piece_type, color)) * value

        return score

    def get_state(self):

        us = self.board.turn
        them = not us

        # Most valuable threatened piece

        max_threatened = 0

        for sq, piece in self.board.piece_map().items():

            if piece.color == us and piece.piece_type != chess.KING:

                if self.board.is_attacked_by(them, sq):

                    value = self.PIECE_VALUES[piece.piece_type]

                    max_threatened = max(max_threatened, value)

        # Best available capture

        best_capture_val = 0
        target_defended = 0

        for move in self.board.legal_moves:

            if self.board.is_capture(move):

                captured_piece = self.board.piece_at(move.to_square)

                if self.board.is_en_passant(move):
                    value = 1
                else:
                    value = self.PIECE_VALUES[captured_piece.piece_type]

                if value > best_capture_val:

                    best_capture_val = value

                    target_defended = int(
                        self.board.is_attacked_by(them, move.to_square)
                    )

        # Material advantage

        diff = self._get_material_score(us) - self._get_material_score(them)

        mat_sign = 1 if diff > 0 else -1 if diff < 0 else 0

        return [max_threatened, best_capture_val, target_defended, mat_sign]

    # ACTION SELECTION
    def _select_move_for_action(self, action):
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        us = self.board.turn
        them = not us

        # Capture highest-value piece

        if action == 0:
            captures = []

            for move in legal_moves:
                if self.board.is_capture(move):
                    target = self.board.piece_at(move.to_square)
                    value = self.PIECE_VALUES[target.piece_type] if target else 1
                    captures.append((value, move))

            if captures:
                captures.sort(key=lambda x: x[0], reverse=True)
                return captures[0][1]

        # ACTION 1: Escape threatened piece

        elif action == 1:
            threatened = []
            for sq, piece in self.board.piece_map().items():
                if piece.color == us and piece.piece_type != chess.KING:
                    if self.board.is_attacked_by(them, sq):
                        threatened.append((self.PIECE_VALUES[piece.piece_type], sq))
            threatened.sort(key=lambda x: x[0], reverse=True)

            for _, sq in threatened:
                escape_moves = [m for m in legal_moves if m.from_square == sq]
                safe_escapes = [
                    m
                    for m in escape_moves
                    if not self.board.is_attacked_by(them, m.to_square)
                ]
                if safe_escapes:
                    return random.choice(safe_escapes)
                if escape_moves:
                    return random.choice(escape_moves)

        # Develop toward center
        elif action == 2:
            center_squares = {
                chess.E4,
                chess.D4,
                chess.E5,
                chess.D5,
                chess.C3,
                chess.F3,
                chess.C6,
                chess.F6,
            }

            safe_moves = [
                m
                for m in legal_moves
                if m.to_square in center_squares
                and not self.board.is_attacked_by(them, m.to_square)
            ]

            if safe_moves:
                return random.choice(safe_moves)

        # Any safe legal move
        safe_moves = [
            m for m in legal_moves if not self.board.is_attacked_by(them, m.to_square)
        ]

        if safe_moves:
            return random.choice(safe_moves)
        return random.choice(legal_moves)

    # ENVIRONMENT STEP
    def step(self, action):

        us = self.board.turn
        them = not us
        if self.board.is_game_over():
            return self.get_state(), 0.0, True, None

        material_before = self._get_material_score(us) - self._get_material_score(them)
        move = self._select_move_for_action(action)

        if move is None:
            return self.get_state(), 0.0, True, None

        self.board.push(move)
        self.move_count += 1
        material_after = self._get_material_score(us) - self._get_material_score(them)
        reward = float(material_after - material_before)
        done = False
        if self.board.is_checkmate():
            reward += 20.0
            done = True

        elif self.board.is_game_over():
            done = True
        elif self.move_count >= self.max_moves:
            done = True

        next_state = self.get_state()
        return next_state, reward, done, move


# SELF-PLAY TRAINING
def train_self_play(env, agent, episodes=500):
    print(f"Starting self-play training: {episodes} episodes")

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        # Separate pending transitions for each color
        last_transition = {chess.WHITE: None, chess.BLACK: None}
        while not done:
            current_player = env.board.turn
            opponent = not current_player
            action = agent.act(state)
            next_state, reward, done, move = env.step(action)
            # Finalize opponent's previous transition
            if last_transition[opponent] is not None:
                prev_s, prev_a, prev_r = last_transition[opponent]
                net_reward = prev_r - reward
                agent.learn(prev_s, prev_a, net_reward, next_state, done)
                last_transition[opponent] = None
            # Finalize current player's transition if terminal
            if done:
                agent.learn(state, action, reward, next_state, True)
            else:
                last_transition[current_player] = (state, action, reward)
            state = next_state
        agent.decay_epsilon()

        if ep % 100 == 0:
            print(
                f"Episode {ep}/{episodes} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"States: {len(agent.q_table)}"
            )
    print("Training completed!")


import time
import chess
import pygame


def watch_self_play(env, agent, delay=0.8):
    """Visualizes the agent playing against itself in a Pygame window."""
    pygame.init()

    board_size = 560
    sq_size = board_size // 8
    screen = pygame.display.set_mode((board_size, board_size))
    pygame.display.set_caption("RL Chess - Self-Play")

    # Colors
    light_sq = (240, 217, 181)
    dark_sq = (181, 136, 99)
    highlight_sq = (205, 210, 106)

    # Unicode chess symbols
    piece_symbols = {
        "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
        "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
    }

    # Try system fonts that support chess Unicode symbols well
    font = pygame.font.SysFont("segoeuisymbol,dejavusans,arial", int(sq_size * 0.75))

    def draw_board(last_move=None):
        for row in range(8):
            for col in range(8):
                sq = chess.square(col, 7 - row)
                color = light_sq if (row + col) % 2 == 0 else dark_sq

                # Highlight last move squares
                if last_move and sq in (last_move.from_square, last_move.to_square):
                    color = highlight_sq

                rect = pygame.Rect(col * sq_size, row * sq_size, sq_size, sq_size)
                pygame.draw.rect(screen, color, rect)

                piece = env.board.piece_at(sq)
                if piece:
                    symbol = piece_symbols[piece.symbol()]
                    text_color = (255, 255, 255) if piece.color == chess.WHITE else (20, 20, 20)
                    # Draw subtle outline for white pieces so they pop on light squares
                    if piece.color == chess.WHITE:
                        outline = font.render(symbol, True, (30, 30, 30))
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            out_rect = outline.get_rect(center=(rect.centerx + dx, rect.centery + dy))
                            screen.blit(outline, out_rect)

                    text_surf = font.render(symbol, True, text_color)
                    text_rect = text_surf.get_rect(center=rect.center)
                    screen.blit(text_surf, text_rect)

        pygame.display.flip()

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    state = env.reset()
    done = False
    last_move = None

    draw_board()
    last_step_time = time.time()

    running = True
    while running:
        # Handle window close event cleanly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not done and (time.time() - last_step_time >= delay):
            current_player = env.board.turn
            action = agent.act(state)
            next_state, reward, done, move = env.step(action)

            if move is not None:
                last_move = move
                player_name = "White" if current_player == chess.WHITE else "Black"
                print(f"{player_name} plays: {move.uci()} | Action: {action} | Reward: {reward}")

            state = next_state
            draw_board(last_move)
            last_step_time = time.time()

            if env.board.is_game_over() or done:
                done = True
                outcome = env.board.outcome()
                result_text = outcome.termination.name if outcome else "Move limit reached"
                pygame.display.set_caption(f"Game Over: {env.board.result()} ({result_text})")
                print(f"\nGAME OVER | Result: {env.board.result()} | {result_text}")

        pygame.time.wait(30)

    agent.epsilon = old_epsilon
    pygame.quit()


# MAIN EXECUTION
if __name__ == "__main__":

    env = ChessTacticalEnv(max_moves=120)
    agent = QLearningAgent(action_size=ChessTacticalEnv.ACTION_SIZE)
    # Load previously trained model if available
    agent.load("chess_q_table.json")
    # Train for 100 games
    train_self_play(env, agent, episodes=100)
    # Save trained Q-table
    agent.save("chess_q_table.json")
    # Watch AI play against itself
    watch_self_play(env, agent, delay=1.0)
