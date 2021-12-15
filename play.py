import agent


def main():
    # cuda not necessary at all
    a = agent.Agent(device="cpu")

    # load the 40 million step model
    a.load_model(40000000)

    # print action tensors of every game_tick, 0.25 fps, and rotate screen
    # a.play(print_action=True, tickrate=0.25, rotate=True)

    # play normally
    a.play()


if __name__ == "__main__":
    main()
