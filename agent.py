from collections import deque, namedtuple
from datetime import datetime
import random
import os
import json
import torch
import model
import snake
import pygame

Transition = namedtuple(
    'Transition', ('before_state', 'action', 'after_state', 'reward', 'terminal'))


class Memory:
    def __init__(self, max_mem):
        self.memory = deque([], maxlen=max_mem)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, e_init=0, e_final=0, e_iter=0, b_size=0, lr=0, gamma=0,
                 max_mem=0, save_model_every=0, save_stats_every=0, device="cpu"):
        self.device = device
        self.model = model.Model().to(device=self.device)
        self.memory = Memory(max_mem=max_mem)
        self.game = snake.game()

        self.current_step = 0
        self.epsilon_initial = e_init
        self.epsilon_final = e_final
        self.epsilon = self.epsilon_initial
        self.epsilon_iterations = e_iter
        self.batch_size = b_size
        self.lr = lr
        self.gamma = gamma
        self.max_memory = max_mem

        self.save_model_every = save_model_every
        self.save_stats_every = save_stats_every
        # stats for json file etc
        self.games_played = 0
        self.food_eaten = 0
        self.total_length = 0
        self.length_this_game = 0
        self.average_length = 0
        self.max_length = 0

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.HuberLoss()
        self.training = False

        for param in self.model.parameters():
            param.requires_grad = True

    # this function handles statistics when games are over, or model is being saved
    def update_statistics(self, reward: int):
        if reward > 0:
            self.food_eaten += 1
            self.length_this_game += 1
        elif reward < 0:
            self.games_played += 1
            self.total_length += self.length_this_game
            # calculate average now
            self.average_length = self.total_length / self.games_played
            # check if this is a new record
            if self.length_this_game > self.max_length:
                self.max_length = self.length_this_game
            # reset current length
            self.length_this_game = len(self.game.player)

    # reset statistics, either between games or between saves while training
    def reset_statistics(self):
        self.games_played = 0
        self.food_eaten = 0
        self.total_length = 0
        self.length_this_game = 0
        self.average_length = 0
        self.max_length = 0

    # idk figure this out
    def calculate_loss(self):
        pass

    # expects (N,3,84,84) uint8 tensor
    def predict(self, s: torch.Tensor) -> torch.Tensor:
        model_input = s.to(dtype=torch.float32, device=self.device)

        ret = torch.squeeze(self.model(model_input)).to(device=self.device)
        return ret

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        r = random.random()
        if r < self.epsilon:
            # return random tensor
            ret = torch.rand(3, dtype=torch.float32)
        else:
            with torch.no_grad():
                ret = self.predict(state)

        # decrease epsilon if we are still above e_final
        if self.epsilon > self.epsilon_final:
            self.epsilon -= (self.epsilon_initial -
                             self.epsilon_final)/self.epsilon_iterations

        return ret

    def get_state(self) -> torch.Tensor:
        out = self.game.get_state_data().to(device=self.device)
        if self.game.current_direction == snake.direction.UP:
            pass
        elif self.game.current_direction == snake.direction.LEFT:
            out = torch.rot90(out, k=3, dims=(2, 3))

        elif self.game.current_direction == snake.direction.DOWN:
            out = torch.rot90(out, k=2, dims=(2, 3))

        else:
            out = torch.rot90(out, k=1, dims=(2, 3))

        return out

    def action_tensor_to_index(self, t: torch.Tensor) -> int:
        return torch.argmax(t).item()

    def index_into_direction(self, index: int) -> snake.direction:

        # if index is 1, just keep going in same direction
        if index == 1:
            return self.game.current_direction
        elif index == 0:
            if self.game.current_direction == snake.direction.UP:
                return snake.direction.LEFT
            elif self.game.current_direction == snake.direction.LEFT:
                return snake.direction.DOWN
            elif self.game.current_direction == snake.direction.DOWN:
                return snake.direction.RIGHT
            else:
                return snake.direction.UP
        else:
            if self.game.current_direction == snake.direction.UP:
                return snake.direction.RIGHT
            elif self.game.current_direction == snake.direction.RIGHT:
                return snake.direction.DOWN
            elif self.game.current_direction == snake.direction.DOWN:
                return snake.direction.LEFT
            else:
                return snake.direction.UP

    def train_step(self):
        self.training_step()
        self.optimizer_step(self.batch_size)

    def play(self, print_action=False, tickrate=20):

        pygame.init()

        play_steps = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.game.new_game()

            play_steps += 1
            reward = self.play_step(print_action)
            self.game.draw_game()
            terminal = False if reward >= 0 else True

            # if we just died, print stats about this run
            if terminal:
                print(f"steps: {play_steps}, length: {self.length_this_game}")
                play_steps = 0
                self.reset_statistics()
            else:
                self.update_statistics(reward)

            self.game.clock.tick(tickrate)

    def train(self, draw=True):

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        print(f"training, start time: {current_time}")

        if draw:
            pygame.init()
        while True:
            if draw:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

            terminal = self.train_step()

            if (draw):
                self.game.draw_game()
            # if we are saving this iteration

            if self.current_step % self.save_stats_every == 0:
                # if we just died, don't "forcefully" end the game
                if not terminal:
                    self.update_statistics(-1)

                # save stats now
                self.save_stats()

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(
                    f"{current_time}: step={self.current_step}, games_played={self.games_played}, food_eaten={self.food_eaten}, " +
                    f"average_length={self.average_length}, max_length={self.max_length}")
                # reset statistics for next cycle
                self.reset_statistics()

                if not terminal:
                    self.game.new_game()

            if self.current_step % self.save_model_every == 0:
                self.save_model()

    def optimizer_step(self, size: int):
        # need to sample from memory, and if memory doesn't contain at least our batch_size
        # we can't really do anything
        if len(self.memory) < size:
            return

        transitions = self.memory.sample(size)

        batch = Transition(*zip(*transitions))

        # before/after are ((1,3,84,84)), so torch.cat gives us ((batch_size, 3, 84, 84))
        before_batch = torch.cat(
            batch.before_state).to_dense().to(device=self.device)
        after_batch = torch.cat(
            batch.after_state).to_dense().to(device=self.device)

        # batch.action is tuple(int) in range of (0,3), so action_batch is [batch_size]
        # however we want to index into before_batch with it, so we "unsqueeze"
        # and add another dimension, so it is [batch_size][1]
        action_batch = torch.Tensor(batch.action).type(torch.int64).unsqueeze(-1).to(
            device=self.device)

        # reward_batch is [batch_size]
        reward_batch = torch.Tensor(batch.reward).to(device=self.device)

        # terminal_batch is [batch_size] of bools, to mask q_after_max at end
        terminal_batch = torch.BoolTensor(
            batch.terminal).to(device=self.device)

        # flip because each "terminal" is true when we DON'T want Q(s',a') in y_target
        terminal_batch = ~terminal_batch

        # q_before is a [batch_size][actions] of our initial predictions, in our case
        # q_before is [batch_size][3], since we have 3 possible actions
        q_before = self.predict(before_batch).to(device=self.device)

        # y_pred is what our model currently tells us about states, however we index
        # into q_before with the action indexes we've _previously_ taken, because
        # we know what the reward of that (state, action) was (Q(s,a)). For pytorch reasons,
        # this is a [batch_size][1], so we squeeze() to get [batch_size]
        y_pred = torch.gather(q_before, dim=1, index=action_batch).squeeze().to(
            device=self.device)

       # calculate our after states
        q_after = self.predict(after_batch).detach().to(
            device=self.device)
        # get the max in every row
        q_after_max, ind = torch.max(q_after, dim=1)
        # mask the values when a (state, action) pair gave us a terminal state.
        # q_after_max is [batch_size]
        q_after_max = terminal_batch * q_after_max

        # if terminal:
        # Q(s,a) = reward
        # else:
        # Q(s,a) = reward + gamma * Q(s', a')
        # this is our target
        y_target = reward_batch + self.gamma * q_after_max

        loss = self.criterion(y_pred, y_target)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss

    # just assumes you're saving models into ./model

    def load_model(self, step):
        folder = "./model"
        model_name = f"{step}_model"
        optim_name = f"{step}_opt"
        params_name = f"{step}_params.json"

        model_file = os.path.join(folder, model_name)
        optim_file = os.path.join(folder, optim_name)
        params_file = os.path.join(folder, params_name)

        if not os.path.exists(model_file) or not os.path.exists(optim_file) or not os.path.exists(params_file):
            print("model files not found.")
            return

        self.model.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.optimizer.load_state_dict(
            torch.load(optim_file, map_location='cpu'))
        with open(params_file, 'r') as f:
            params = json.load(f)
            for key in params:
                setattr(self, key, params[key])

        self.memory = Memory(self.max_memory)

    def save_stats(self):
        folder = "./model/stats"
        if not os.path.exists(folder):
            os.makedirs(folder)

        stats = {}
        stats['step'] = self.current_step
        stats['games_played'] = self.games_played
        stats['food_eaten'] = self.food_eaten
        stats['total_length'] = self.total_length
        stats['average_length'] = self.average_length
        stats['max_length'] = self.max_length

        stats_name = f"{self.current_step}_stats.json"

        stats_file = os.path.join(folder, stats_name)

        with open(stats_file, "w") as s:
            json.dump(stats, s)

    def save_model(self):
        folder = "./model"

        if not os.path.exists(folder):
            os.makedirs(folder)

        model_name = f"{self.current_step}_model"
        optim_name = f"{self.current_step}_opt"
        params_name = f"{self.current_step}_params.json"

        model_file = os.path.join(folder, model_name)
        torch.save(self.model.state_dict(), model_file)

        optim_file = os.path.join(folder, optim_name)
        torch.save(self.optimizer.state_dict(), optim_file)

        params = {}
        params['gamma'] = self.gamma
        params['current_step'] = self.current_step
        params['epsilon_initial'] = self.epsilon_initial
        params['epsilon_final'] = self.epsilon_final
        params['epsilon'] = self.epsilon
        params['epsilon_iterations'] = self.epsilon_iterations
        params['lr'] = self.lr
        params['batch_size'] = self.batch_size
        params['max_memory'] = self.max_memory
        params['save_model_every'] = self.save_model_every
        params['save_stats_every'] = self.save_stats_every

        params_file = os.path.join(folder, params_name)
        with open(params_file, "w") as out:
            json.dump(params, out)

    # play step will just infer best move and act accordingly, no memory or learning
    def play_step(self, print_action):
        # get our state
        state_tensor = self.get_state()

        action_tensor = self.predict(state_tensor)
        action_index = self.action_tensor_to_index(action_tensor)
        # determine our action
        action = self.index_into_direction(action_index)

        if print_action:
            if action_index == 0:
                action_str = "turn left"
            elif action_index == 1:
                action_str = "go straight"
            else:
                action_str = "turn right"

            print("[{:.3f},{:.3f},{:.3f}] -> {}".format(action_tensor[0],
                  action_tensor[1], action_tensor[2], action_str))
        # perform action
        return self.game.game_tick(action)

    def training_step(self):
        self.current_step += 1

        # transform game state into a tensor with a consistent orientation
        before_state_tensor = self.get_state()

        # get our action tensor, either random tensor or output from model
        action_tensor = self.get_action(before_state_tensor)

        # convert action_tensor into an action we can actually do in the game
        action_index = self.action_tensor_to_index(action_tensor)
        action = self.index_into_direction(action_index)

        # step once in the game, and get our reward
        reward = self.game.game_tick(action)

        # get next state tensor
        after_state_tensor = self.get_state()

        terminal = True if reward < 0 else False

        # save into our replay memory, for further training
        self.memory.push(before_state_tensor.to_sparse(), action_index,
                         after_state_tensor.to_sparse(), reward, terminal)

        # update our statistics
        self.update_statistics(reward)

        return terminal
