import pygame
import sys
import random
from enum import Enum
import numpy as np

# globals because lazy
# game is 32 x 32 rectangles
size = width, height = 800, 800
rect_size = 25
grid_size = int(width / rect_size)

black = 0, 0, 0
green = 0, 255, 0
yellow = 255, 255, 0


class direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class cell(Enum):
    BG = 0
    SNAKE = 1
    HEAD = 2
    FOOD = 3


class rect:
    x = 0
    y = 0

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


class player:
    body = []

    def __init__(self):
        self.restart()

    def restart(self):
        self.body = [rect(16, 16), rect(16, 15), rect(16, 14)]

    # head is always first element of body
    def head(self) -> rect:
        return self.body[0]

    def process_move(self, move: direction):

        head = self.body[0]

        if move == direction.UP:
            self.body.insert(0, rect(head.x, head.y - 1))
        elif move == direction.DOWN:
            self.body.insert(0, rect(head.x, head.y + 1))
        elif move == direction.LEFT:
            self.body.insert(0, rect(head.x - 1, head.y))
        else:
            self.body.insert(0, rect(head.x + 1, head.y))


class game:
    state = np.zeros((grid_size, grid_size), dtype=int)
    food = rect(0, 0)
    screen = pygame.display.set_mode(size)
    player = player()
    current_direction = direction.RIGHT
    clock = pygame.time.Clock()

    def draw_game(self):
        self.screen.fill(black)
        for x in range(grid_size):
            for y in range(grid_size):
                color = 0, 0, 0
                if self.state[x][y] == cell.BG.value:
                    color = black
                elif self.state[x][y] == cell.SNAKE.value or self.state[x][y] == cell.HEAD.value:
                    color = green
                else:  # food case
                    color = yellow
                pygame.draw.rect(self.screen, color, pygame.Rect(
                    x*rect_size, y*rect_size, rect_size, rect_size))

        pygame.display.flip()

    def game_tick(self, action: direction):
        next_direction = action
        # make sure we aren't turning around
        if next_direction == direction.UP and self.current_direction != direction.DOWN:
            self.current_direction = direction.UP
        if next_direction == direction.DOWN and self.current_direction != direction.UP:
            self.current_direction = direction.DOWN
        if next_direction == direction.LEFT and self.current_direction != direction.RIGHT:
            self.current_direction = direction.LEFT
        if next_direction == direction.RIGHT and self.current_direction != direction.LEFT:
            self.current_direction = direction.RIGHT

        self.player.process_move(self.current_direction)
        # check for loss cases, either player has run into a wall, or themselves
        head = self.player.head()
        game_over = False
        eat = False

        # if we've run into wall
        if head.x > grid_size - 1 or head.x < 0 or head.y > grid_size - 1 or head.y < 0:
            game_over = True

        # if we've run into ourself
        for part in self.player.body[1:]:
            if head == part:
                game_over = True

        if head == self.food:
            eat = True

        # new game if we've died
        if game_over:
            self.new_game()
        else:

            if eat:
                self.new_food()
            else:
                # if we don't eat, we must pop off player.body
                self.player.body.pop()

        self.fill_state()

        # return a tuple of state, whether the game just ended, and if we ate food
        return (self.state, game_over, eat)

    def fill_state(self):
        self.state.fill(0)

        for part in self.player.body:
            self.state[part.x][part.y] = cell.SNAKE.value

        head = self.player.head()
        self.state[head.x][head.y] = cell.HEAD.value

        self.state[self.food.x][self.food.y] = cell.FOOD.value

    def play_game_tick(self):
        # get input
        next_direction = self.current_direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    next_direction = direction.UP
                elif event.key == pygame.K_a:
                    next_direction = direction.LEFT
                elif event.key == pygame.K_s:
                    next_direction = direction.DOWN
                elif event.key == pygame.K_e:
                    next_direction = direction.RIGHT

        self.game_tick(next_direction)

        self.draw_game()

        self.clock.tick(20)

    def new_game(self):
        self.player.restart()
        self.new_food()
        self.current_direction = direction.RIGHT

    def new_food(self):
        self.food = rect(random.randint(0, grid_size - 1),
                         random.randint(0, grid_size - 1))
        # check to make sure our snake isn't on this point
        for part in self.player.body:
            if part == self.food:
                self.new_food()
                return


def main():
    pygame.init()
    g = game()

    while True:
        g.play_game_tick()


if __name__ == "__main__":
    main()
