import agent
import pygame
import torch


def main():
    a = agent.Agent()

    pygame.init()
    print(pygame.display.list_modes())
    torch.autograd.set_detect_anomaly(True)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

        a.train_step()
        a.game.draw_game()
        if a.current_step % 10000 == 0:
            a.model.save("model")
            print(str(a.current_step) + ": saving")


if __name__ == "__main__":
    main()
