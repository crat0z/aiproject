# import pygame
import random
from enum import Enum
import torch

# globals because lazy
# game is 32 x 32 rectangles
size = width, height = 1260, 1260
rect_size = 15
grid_size = int(width / rect_size)

black = 0, 0, 0
green = 157, 255, 100
head = 0, 255, 0
yellow = 255, 255, 0

food_start = 50


class direction(Enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


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

    def __len__(self):
        return len(self.body)

    def __init__(self):
        self.restart()

    def restart(self):
        x = int(grid_size/2)
        y = int(grid_size/2)
        self.body = [rect(x, y), rect(x, y-1), rect(x, y-2),
                     rect(x, y-3), rect(x, y-4)]

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
    def __init__(self):
        # use np.uint8 for size
        self.state = torch.zeros((grid_size, grid_size), dtype=torch.uint8)
        self.player = player()
        self.current_direction = direction.DOWN

        self.food_list = []
        # call new_game() so state is not zero
        self.new_game()
        self.drawing = False

    def generate_food(self):
        for i in range(food_start):
            self.new_food()

    def get_state_data(self):
        ret = torch.zeros((1, 3, 84, 84), dtype=torch.uint8)

        # add snake info
        for part in self.player.body:
            ret[0][0][part.x][part.y] = 1

        # add head layer
        head = self.player.head()
        ret[0][1][head.x][head.y] = 1

        # add food
        for f in self.food_list:
            ret[0][2][f.x][f.y] = 1

        return ret

    def rotate_view(self):
        # i may have gotten the rotations wrong for the training..
        # nonetheless, the model still performs well. and not enough time to retrain.
        if self.current_direction == direction.UP:
            pass
        elif self.current_direction == direction.LEFT:
            self.state = torch.rot90(self.state, k=1)

        elif self.current_direction == direction.DOWN:
            self.state = torch.rot90(self.state, k=2)

        else:
            self.state = torch.rot90(self.state, k=3)

    def draw_game(self, rotate):
        if self.drawing == False:
            self.drawing = True

            import pygame

            self.screen = pygame.display.set_mode(size)
            self.clock = pygame.time.Clock()

        self.fill_state()

        if rotate:
            self.rotate_view()

        self.screen.fill(black)
        for x in range(grid_size):
            for y in range(grid_size):
                color = 0, 0, 0
                if self.state[x][y] == cell.BG.value:
                    color = black
                elif self.state[x][y] == cell.SNAKE.value:
                    color = green
                elif self.state[x][y] == cell.HEAD.value:
                    color = head
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

        # check to see if we're eating something this tick
        if head in self.food_list:
            self.food_list.remove(head)
            eat = True
            if not self.food_list:
                self.generate_food()

        # if we don't eat, we must pop off player.body
        if not eat:
            self.player.body.pop()

        # if we've run into wall
        if head.x > grid_size - 1 or head.x < 0 or head.y > grid_size - 1 or head.y < 0:
            game_over = True

        # if we've run into ourself
        for part in self.player.body[1:]:
            if head == part:
                game_over = True

        # calculate reward
        if game_over:
            reward = -1
            self.new_game()
        elif eat:
            reward = 1
        else:
            reward = 0

        return reward

    def fill_state(self):
        self.state.fill_(cell.BG.value)

        for food in self.food_list:
            self.state[food.x][food.y] = cell.FOOD.value

        for part in self.player.body:
            self.state[part.x][part.y] = cell.SNAKE.value

        head = self.player.head()
        self.state[head.x][head.y] = cell.HEAD.value

    def new_game(self):
        self.player.restart()
        self.food_list.clear()
        self.generate_food()
        self.current_direction = direction.DOWN
        self.fill_state()

    def new_food(self):

        while True:
            done = True
            location = rect(random.randint(0, grid_size - 1),
                            random.randint(0, grid_size - 1))
            for part in self.player.body:
                if part == location:
                    done = False
                    break
            if done:
                for part in self.food_list:
                    if part == location:
                        done = False
                        break

            if done:
                self.food_list.append(location)
                return
