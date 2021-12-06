from collections import deque, namedtuple
import random

import torch
import model
import snake

# hyperparameters
max_memory = 10000
# initial epsilon, if rand() < epsilon, a random action is taken
e_initial = 1.0
# final epsilon value
e_final = 0.05
# number of iterations between e_initial and e_end
e_iterations = 100000
# discounted
gamma = 0.95
# memory batch size
batch_size = 128
# learning rate
learning_rate = 0.00002

# epsilon difference per iteration
e_diff = (e_initial - e_final)/e_iterations

Transition = namedtuple(
    'Transition', ('before_state', 'action', 'after_state', 'reward', 'terminal'))

device = torch.device('cuda')


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
        ret = torch.zeros(4, device=device)
        if r < self.epsilon:
            # return random tensor
            ret = torch.rand(4).to(device=device)
        else:
            ret = self.model(state).to(device=device)

        # decrease epsilon if we are still above e_final
        if self.epsilon > e_final:
            self.epsilon -= e_diff

        return ret

    def tensor_to_action(self, t: torch.Tensor) -> snake.direction:
        index = torch.argmax(t)
        if index == 0:
            return snake.direction.UP
        elif index == 1:
            return snake.direction.LEFT
        elif index == 2:
            return snake.direction.DOWN
        else:
            return snake.direction.RIGHT

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
        # state as a pytorch tensor
        before_state = torch.from_numpy(
            self.game.state.flatten()).to(device=device)

        # get our action tensor, either random 4 tensor or output from model
        action_tensor = self.get_action(before_state).to(device=device)

        # convert action_tensor into an action we can actually do in the game
        action = self.tensor_to_action(action_tensor)
        # step once in the game, and get our reward
        reward = self.game.game_tick(action)
        # get after state
        after_state = torch.from_numpy(
            self.game.state.flatten()).to(device=device)

        terminal = True if reward == -5 else False

        td = Transition(before_state=before_state, action=action_tensor,
                        after_state=after_state, reward=reward, terminal=terminal)

        self.memory.push(td)
