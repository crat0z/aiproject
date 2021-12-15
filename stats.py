import matplotlib.pyplot as plt
import numpy as np
import os
import json


def average_len_plot(steps: np.ndarray, avg_len: np.ndarray):
    fig, ax = plt.subplots()

    m, b = np.polyfit(x=steps, y=avg_len, deg=1)

    f_x = "{:e}x + {:.4f}".format(m, b)

    ax.plot(steps, avg_len, label="average length")
    ax.plot(steps, m * steps + b, label=f_x)

    ax.set_xlabel("steps")
    ax.set_ylabel("average length")
    ax.set_title("average length")
    ax.axvline(x=1000000, color='r', linestyle='--')

    ax.legend()


def max_len_plot(steps: np.ndarray, max_len: np.ndarray):
    fig, ax = plt.subplots()

    m, b = np.polyfit(x=steps, y=max_len, deg=1)

    f_x = "{:e}x + {:.4f}".format(m, b)

    ax.plot(steps, max_len, label="max length")
    ax.plot(steps, m * steps + b, label=f_x)

    ax.set_xlabel("steps")
    ax.set_ylabel("max length")
    ax.set_title("max length")
    ax.axvline(x=1000000, color='r', linestyle='--')

    ax.legend()


def games_played_plot(steps: np.ndarray, games_played: np.ndarray, food_eaten: np.ndarray):
    fig, ax = plt.subplots()

    ax.plot(steps, games_played, label="punishments")
    ax.plot(steps, food_eaten, label="rewards")
    ax.set_xlabel("steps")
    ax.set_title("rewards and punishments")
    ax.axvline(x=1000000, color='r', linestyle='--')
    ax.legend()


def main():
    steps = []

    games_played = []
    food_eaten = []
    average_len = []
    max_len = []

    for filename in os.listdir("./model/stats"):
        path = os.path.join("./model/stats/", filename)
        with open(path, "r") as f:
            data = json.load(f)
            steps.append(data['step'])
            games_played.append(data['games_played'])
            food_eaten.append(data['food_eaten'])
            average_len.append(data['average_length'])
            max_len.append(data['max_length'])

    steps = np.array(steps)
    games_played = np.array(games_played)
    food_eaten = np.array(food_eaten)
    average_len = np.array(average_len)
    max_len = np.array(max_len)

    average_len_plot(steps, average_len)

    max_len_plot(steps, max_len)

    games_played_plot(steps, games_played, food_eaten)

    plt.show()


if __name__ == "__main__":
    main()
