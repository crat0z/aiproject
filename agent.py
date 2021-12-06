from collections import deque, namedtuple
import random
import numpy as np

import torch
import model
import snake

# hyperparameters
max_memory = 10000
# initial epsilon, if rand() < epsilon, a random action is taken
e_initial = 0.75
# final epsilon value
e_final = 0.02
# number of iterations between e_initial and e_end
e_iterations = 10000
# discounted
gamma = 0.95
# memory batch size
batch_size = 64
# learning rate
learning_rate = 0.0001

# epsilon difference per iteration
e_diff = (e_initial - e_final)/e_iterations

Transition = namedtuple(
    'Transition', ('before_state', 'action', 'after_state', 'reward', 'terminal'))

device = torch.device('cpu')


class Memory:
    def __init__(self):
        self.memory = deque([], maxlen=max_memory)

    def push(self, t):
        self.memory.append(t)

    def sample(self):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self):
        self.model = model.Model().to(device=device)
        self.memory = Memory()
        self.game = snake.game()
        self.current_step = 0
        self.epsilon = e_initial
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

        for param in self.model.parameters():
            param.requires_grad = True

    def get_action(self, state) -> torch.Tensor:
        r = random.random()
        if r < self.epsilon:
            # return random tensor
            ret = torch.rand(3).to(device=device)
        else:
            ret = self.model(state).to(device=device)

        # decrease epsilon if we are still above e_final
        if self.epsilon > e_final:
            self.epsilon -= e_diff

        return ret

    # snake player can never turn around, so i've eliminated that from output.
    # depending on the orientation of the snake's head, we can rotate the matrix
    # and maybe it'll perform better?
    """ >> > x
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        >> > x[0][1] = 5
        >> > np.rot90(x, 0)
        array([[0., 5., 0.],
               [0., 0., 0.],
               [0., 0., 0.]])
        >> > np.rot90(x, 1)
        array([[0., 0., 0.],
               [5., 0., 0.],
               [0., 0., 0.]])
        >> > np.rot90(x, 2)
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 5., 0.]])
        >> > np.rot90(x, 3)
        array([[0., 0., 0.],
               [0., 0., 5.],
               [0., 0., 0.]]) """

    def normalize_np_into_tensor(self, input: np.ndarray, direction: snake.direction) -> torch.Tensor:
        if direction == snake.direction.UP:
            return torch.from_numpy(input.flatten()).to(device=device)
        elif direction == snake.direction.RIGHT:
            return torch.from_numpy(np.rot90(input, 1).flatten()).to(device=device)
        elif direction == snake.direction.DOWN:
            return torch.from_numpy(np.rot90(input, 2).flatten()).to(device=device)
        else:
            return torch.from_numpy(np.rot90(input, 3).flatten()).to(device=device)

    def tensor_to_action(self, t: torch.Tensor, current_direction: snake.direction) -> snake.direction:
        index = torch.argmax(t)
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

    def optimizer_step(self):
        # need to sample from memory, and if memory doesn't contain at least our batch_size
        # we can't really do anything
        if len(self.memory) < batch_size:
            return

        # random sample from memory
        before, action, after, reward, terminal = zip(*self.memory.sample())

        before_state_batch = torch.stack(before).to(device=device)
        action_batch = torch.stack(action).to(device=device)
        after_state_batch = torch.stack(after).to(device=device)
        reward_batch = torch.tensor(reward).to(device=device)
        terminal_batch = torch.tensor(terminal).to(device=device)

        # q_before is [batch_size][output_size]
        q_before = self.model(before_state_batch).to(device=device)

        y_batch = q_before.clone()
        for i in range(batch_size):
            y_batch[i][torch.argmax(action_batch[i]).item(
            )] = reward_batch[i] if terminal_batch[i] else reward_batch[i] + self.gamma * torch.max(self.model(after_state_batch[i]))

        self.optimizer.zero_grad()

        loss = self.criterion(q_before, y_batch)
        loss.backward()

        self.optimizer.step()

    def step(self):
        self.current_step += 1

        # get the array, current direction
        before_state_np = self.game.state
        before_direction = self.game.current_direction

        # transform it into a tensor which a consistent orientation
        before_state_tensor = self.normalize_np_into_tensor(
            before_state_np, before_direction)

        # get our action tensor, either random tensor or output from model
        action_tensor = self.get_action(before_state_tensor)

        # convert action_tensor into an action we can actually do in the game
        action = self.tensor_to_action(action_tensor, before_direction)

        # step once in the game, and get our reward
        reward = self.game.game_tick(action)

        # get after state
        after_state_np = self.game.state
        after_direction = self.game.current_direction
        after_state_tensor = self.normalize_np_into_tensor(
            after_state_np, after_direction)

        terminal = True if reward == -5 else False

        td = Transition(before_state=before_state_tensor, action=action_tensor,
                        after_state=after_state_tensor, reward=reward, terminal=terminal)

        self.memory.push(td)
