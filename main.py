import numpy
import agent
from datetime import datetime
# hyperparameters
max_memory = 50000
# initial epsilon, if rand() < epsilon, a random action is taken
e_initial = 0.99
# final epsilon value
e_final = 0.1
# number of iterations between e_initial and e_end
e_iterations = 1000000
# discount factor
gamma = 0.99
# memory batch size
batch_size = 32
# learning rate
learning_rate = 0.000002
# save every 5000 steps
save_every = 100000
# change to cpu if your gpu doesnt support cuda
device = "cpu"


def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print(f"start time: {current_time}")

    # start training a new model with the above hyperparameters
    a = agent.Agent(e_initial, e_final, e_iterations,
                    batch_size, learning_rate, gamma, max_memory, save_every, device)

    #a = agent.Agent()
    # a.load(3700000)
    # a.play()
    a.train(draw=False)


if __name__ == "__main__":
    main()
