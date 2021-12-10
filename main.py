import numpy
import agent
from datetime import datetime
# hyperparameters
max_memory = 125000
# initial epsilon, if rand() < epsilon, a random action is taken
e_initial = 0.99
# final epsilon value
e_final = 0.1
# number of iterations between e_initial and e_end
e_iterations = 500000
# discount factor
gamma = 0.99
# memory batch size
batch_size = 32
# learning rate
learning_rate = 0.00002
# save model/stats
save_model_every = 100000
save_stats_every = 25000
# change to cpu if your gpu doesnt support cuda
device = "cuda"


def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print(f"start time: {current_time}")

    # start training a new model with the above hyperparameters
    a = agent.Agent(e_initial, e_final, e_iterations,
                    batch_size, learning_rate, gamma,
                    max_memory, save_model_every, save_stats_every, device)

    #a = agent.Agent()
    # a.load_model(400000)
    # a.play()
    a.train(draw=False)


if __name__ == "__main__":
    main()
