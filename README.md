# AI project
This is the source code for my project in an undergraduate artificial intelligence course in fall 2021. For my project, I trained a reinforcement learning agent in a variant of the game _Snake_. It uses a convolutional neural network similar to the CNN described in _Human-level control through deep reinforcement learning_. The algorithm used to train the neural network is Deep Q Learning. A project report was required for this assignment however it is not included in this repository. For this project, I received a grade of 98%.

## trained models included
For the project, I trained for 40 million steps. The hyperparameters used are the same ones in _train.py_. In the `model` directory, there are 4 saved checkpoints:
* _100k_: the first saved step when I began training.
* _1m_: epsilon has finished decreasing, and is now 0.1.
* _15m_: Judging by the graphs, training became much slower after this point
* _40m_: the final product

## files included
* _gameplay_example.mp4_: a 10 minute clip of normal gameplay, normal speed.
* _gameplay_example_rotating_actions.mp4_: another 10 minute clip, however this time the action tensor at every step is printed to the console, and the screen is rotated to give a better representation of what the agent “sees”. The gameplay is also slowed down, for the viewer’s ease of watching.
* _stats.py_: script used to generate the graphs from the stats collected while training. Requires matplotlib, and numpy.
* _train.py_: script to start training. Requires PyTorch, and Pygame if you wish to watch it while it trains.
* _play.py_: script to watch an agent play. Requires PyTorch and Pygame.
* _model.py_: defines the neural network for the model.
* _agent.py_: the implementation of the agent class. Contains the vast majority of the code.
* _snake.py_: contains the implementation of the snake game.

