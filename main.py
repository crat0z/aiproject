import agent
from datetime import datetime
# hyperparameters
max_memory = 10000
# initial epsilon, if rand() < epsilon, a random action is taken
e_initial = 0.99
# final epsilon value
e_final = 0.05
# number of iterations between e_initial and e_end
e_iterations = 50000
# discount factor
gamma = 0.99
# memory batch size
batch_size = 32
# learning rate
learning_rate = 0.0001
# save every 5000 steps
save_every = 5000


def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print(f"start time: {current_time}")

    # start training a new model with the above hyperparameters
    a = agent.Agent(e_initial, e_final, e_iterations,
                    batch_size, learning_rate, gamma, max_memory, save_every)

    # or load a previously trained one
    # a = agent.Agent()
    # a.load(1710000)
    a.train(draw=False)


if __name__ == "__main__":
    main()
