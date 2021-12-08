from collections import deque, namedtuple
import random
import numpy as np
import os
import json
import torch
import model
import snake
import pygame

Transition = namedtuple(
    'Transition', ('before_state', 'action', 'after_state', 'reward', 'terminal'))

device = torch.device('cuda')


class Memory:
    def __init__(self, max_mem):
        self.memory = deque([], maxlen=max_mem)

    def push(self, before, act, after, rew, term):
        self.memory.append(Transition(before, act, after, rew, term))

    # not zipping this right here is slower
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self, e_init=0, e_final=0, e_iter=0, b_size=0, lr=0, gamma=0, max_mem=0, save_every=0):
        self.model = model.Model().to(device=device)
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

        self.save_every = save_every

        # stats for json file etc
        self.games_played = 0
        self.food_eaten = 0
        self.total_length = 0
        self.length_this_game = 0
        self.average_length = 0
        self.max_length = 0

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()

        for param in self.model.parameters():
            param.requires_grad = True

    def get_action(self, state) -> torch.Tensor:
        r = random.random()
        if r < self.epsilon:
            # return random tensor
            ret = torch.rand(3, dtype=torch.float32).to(device=device)
        else:
            with torch.no_grad():
                ret = torch.squeeze(self.model(state)).to(device=device)

        # decrease epsilon if we are still above e_final
        if self.epsilon > self.epsilon_final:
            self.epsilon -= (self.epsilon_initial -
                             self.epsilon_final)/self.epsilon_iterations

        return ret

    def normalize_np_into_tensor(self, input: np.ndarray) -> torch.Tensor:
        # maybe this is better, idk
        input = np.reshape(input, (1, 1, 15, 15))
        return torch.from_numpy(input).to(device=device)

    def action_tensor_to_index(self, t: torch.Tensor) -> int:
        return torch.argmax(t).item()

    def tensor_to_action(self, t: torch.Tensor, current_direction: snake.direction) -> snake.direction:
        index = self.action_tensor_to_index(t)
        # if arg == 1, then just return current_direction
        if index == 1:
            return current_direction
        else:
            # i wish python had switch statements
            # if we're going left
            if index == 0:
                if current_direction == snake.direction.UP:
                    return snake.direction.LEFT
                if current_direction == snake.direction.DOWN:
                    return snake.direction.RIGHT
                if current_direction == snake.direction.LEFT:
                    return snake.direction.DOWN
                if current_direction == snake.direction.RIGHT:
                    return snake.direction.UP
            else:  # if we're going right
                if current_direction == snake.direction.UP:
                    return snake.direction.RIGHT
                if current_direction == snake.direction.DOWN:
                    return snake.direction.LEFT
                if current_direction == snake.direction.LEFT:
                    return snake.direction.UP
                if current_direction == snake.direction.RIGHT:
                    return snake.direction.DOWN

    def train_step(self):
        self.step()
        self.optimizer_step()

    def train(self, draw=True):
        if draw:
            pygame.init()
        while True:
            if draw:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

            self.train_step()

            if (draw):
                self.game.draw_game()
            if self.current_step % self.save_every == 0:
                self.save()
                print(
                    f"step {self.current_step}: games_played = {self.games_played}, food_eaten = {self.food_eaten}, " +
                    f"average_length = {self.average_length}, max_length = {self.max_length}")

    def optimizer_step(self):
        # need to sample from memory, and if memory doesn't contain at least our batch_size
        # we can't really do anything
        if len(self.memory) < self.batch_size:
            return

        # random sample from memory
        before, action, after, reward, terminal = self.memory.sample(
            self.batch_size)

        # before,after are [batch_size][input_size]
        before_state_batch = torch.stack(before).to(device=device)
        after_state_batch = torch.stack(after).to(device=device)

        #action is [batch_size][output_size]
        action_batch = torch.stack(action).to(device=device)

        # reward,terminal are [batch_size]
        reward_batch = torch.tensor(reward).to(device=device)
        terminal_batch = torch.tensor(terminal).to(device=device)

        # q_before/q_after is [batch_size][output_size]
        q_before = torch.zeros((self.batch_size, 3))
        q_after = torch.zeros((self.batch_size, 3))

        for i in range(self.batch_size):
            q_before[i] = self.model(before_state_batch[i]).squeeze()
        with torch.no_grad():
            for i in range(self.batch_size):
                q_after[i] = self.model(after_state_batch[i]).squeeze()

        # y_batch represents our target
        y_batch = q_before.clone()
        # Q(s,a) = r(s, a) + gamma * Q(s', a')
        for i in range(self.batch_size):
            y_batch[i][self.action_tensor_to_index(
                action_batch[i])] = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * torch.max(q_after[i])

        loss = self.criterion(y_batch, q_before)

        self.optimizer.zero_grad()
        # calculate the loss (MSE in this case) between our current prediction q_before
        # and our target y_batch
        loss.backward()

        self.optimizer.step()

    # just assumes you're saving models into ./model
    def load(self, step):
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

        self.model.load_state_dict(torch.load(model_file))
        self.optimizer.load_state_dict(torch.load(optim_file))
        with open(params_file, 'r') as f:
            params = json.load(f)
            for key in params:
                setattr(self, key, params[key])

        self.memory = Memory(self.max_memory)

    def save(self):
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
        params['save_every'] = self.save_every

        params['games_played'] = self.games_played
        params['food_eaten'] = self.food_eaten

        params['total_length'] = self.total_length
        params['average_length'] = self.average_length
        params['max_length'] = self.max_length

        params_file = os.path.join(folder, params_name)
        with open(params_file, "w") as out:
            json.dump(params, out)

    def step(self):
        self.current_step += 1

        before_direction = self.game.current_direction

        # transform game state into a tensor with a consistent orientation
        before_state_tensor = self.normalize_np_into_tensor(
            self.game.state)

        # get our action tensor, either random tensor or output from model
        action_tensor = self.get_action(before_state_tensor)

        # convert action_tensor into an action we can actually do in the game
        action = self.tensor_to_action(action_tensor, before_direction)

        # step once in the game, and get our reward
        reward = self.game.game_tick(action)

        # get next state tensor
        after_state_tensor = self.normalize_np_into_tensor(
            self.game.state)

        terminal = True if reward < 0 else False

        # we've eaten food this iteration
        if reward > 0:
            terminal = False
            self.food_eaten += 1
            self.length_this_game += 1
        elif reward < 0:
            terminal = True
            self.games_played += 1
            self.total_length += self.length_this_game
            # calculate average now
            self.average_length = self.total_length / self.games_played
            # check if this is a new record
            if self.length_this_game > self.max_length:
                self.max_length = self.length_this_game
            # reset current length
            self.length_this_game = len(self.game.player)
        else:
            terminal = False

        self.memory.push(before_state_tensor, action_tensor,
                         after_state_tensor, reward, terminal)
