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
e_final = 0.01
# number of iterations between e_initial and e_end
e_iterations = 10000
# some value in bellman equation??
gamma = 0.90
# memory batch size
batch_size = 128
# learning rate
learning_rate = 0.00005

# epsilon difference per iteration
e_diff = (e_initial - e_final)/e_iterations

Transition = namedtuple(
    'Transition', ('before_state', 'action', 'after_state', 'reward'))


class memory:
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
        self.model = model.Model().to("cpu")
        self.memory = memory()
        self.game = snake.game()
        self.current_step = 0
        self.epsilon = e_initial
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def get_action(self, state) -> snake.direction:
        r = random.random()
        # just take random action if r < epsilon
        if r < self.epsilon:
            r = random.randint(0, 3)
            if r == 0:
                return snake.direction.UP
            elif r == 1:
                return snake.direction.LEFT
            elif r == 2:
                return snake.direction.DOWN
            else:
                return snake.direction.RIGHT
        else:
            # otherwise ask the model
            pass

        # decrease epsilon if we are still above e_final
        if self.epsilon > e_final:
            self.epsilon -= e_diff

    def step(self):
        self.current_step += 1
        # state as a pytorch tensor
        before_state = torch.from_numpy(self.game.state.flatten())
        action = self.get_action(before_state)
        reward = self.game.game_tick(action)
        after_state = torch.from_numpy(self.game.state.flatten())

        td = Transition(before_state=before_state, action=action,
                        after_state=after_state, reward=reward)

        self.memory.push(td)
