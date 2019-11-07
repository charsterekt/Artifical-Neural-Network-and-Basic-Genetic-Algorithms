import pyglet
from pyglet.window import key
from pyglet.graphics import Batch
from pyglet.image import SolidColorImagePattern
from pyglet.window import Window as OriginalWindow
from pyglet.sprite import Sprite as OriginalSprite
from pyglet.text import Label
from typing import NamedTuple, Tuple, Callable, Optional, Union
import random
from time import sleep


class Position(NamedTuple):
    x: int
    y: int


class Window(OriginalWindow):
    def get_size(self):
        return Position(*super().get_size())


class Sprite(OriginalSprite):

    @OriginalSprite.position.getter
    def position(self):
        return Position(self._x, self._y)

    def move(self, dx: int = 0, dy: int = 0):
        self.update(x=self.x + dx, y=self.y + dy)

    def bounding_box(self) -> Tuple[Position, Position]:
        return (
            Position(
                self.x - self.image.anchor_x,
                self.y - self.image.anchor_y,
            ),
            Position(
                self.x - self.image.anchor_x + self.width,
                self.y - self.image.anchor_y + self.height,
            ),
        )

    def overlaps(self, other: Union["Sprite", Window], *, fully: bool = False):
        p1, p2 = self.bounding_box()
        if isinstance(other, Sprite):
            p3, p4 = other.bounding_box()
        else:
            p3, p4 = Position(0, 0), other.get_size()
        if p2.x > p3.x and p4.x > p1.x and p2.y > p3.y and p4.y > p1.y:
            if fully:
                if (p2.x >= p4.x and p2.y >= p4.y and p3.x >= p1.x and p3.y >= p1.y
                   or p4.x >= p2.x and p4.y >= p2.y and p1.x >= p3.x and p1.y >= p3.y):
                    return True
                return False
            return True
        return False


def center(image):
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2


apple_pic = SolidColorImagePattern((0, 255, 0, 255)).create_image(10, 10)
snake_seg = SolidColorImagePattern((255, 0, 0, 255)).create_image(20, 20)
center(apple_pic)
center(snake_seg)


class Game:
    def __init__(self):
        # If you make changes here, then make changes in the reset function up ahead as well
        self.event_loop = pyglet.app.EventLoop()
        pyglet.app.event_loop = self.event_loop
        self.clock = pyglet.clock.Clock()
        self.event_loop.clock = self.clock
        self.window = Window(600, 600, caption="Snake", vsync=False)
        self.main_batch = Batch()

        self.snake = [
            Sprite(snake_seg, 290, 290, batch=self.main_batch),
            Sprite(snake_seg, 270, 290, batch=self.main_batch),
            Sprite(snake_seg, 250, 290, batch=self.main_batch),
            Sprite(snake_seg, 230, 290, batch=self.main_batch),
            Sprite(snake_seg, 210, 290, batch=self.main_batch),
        ]
        self.head = self.snake[0]  # first segment of the snake
        self.apple = Sprite(apple_pic, random.randint(10, 590), random.randint(10, 590), batch=self.main_batch)
        self.gen_apple()  # ensures the apple doesn't overlap with the snake
        self.label = Label("Score: 0", x=10, y=580, batch=self.main_batch)
        self.window.event(self.on_draw)

        self.score = 0
        self.steps = 0
        self.turns = 0  # Not currently in use
        self.exit_code = 0
        self.direction = 0
        self.current_direction = 0
        self._fps = 0
        self._external = None
        self.fps = 10
        self.external = None

    def exit(self):
        self.event_loop.exit()
        self.window.close()

    def run(self):
        self.event_loop.run()

    def on_draw(self):
        self.window.clear()
        self.main_batch.draw()

    def gen_apple(self):
        while True:
            self.apple.update(x=random.randint(10, 590), y=random.randint(10, 590))
            for segment in self.snake:
                if segment.overlaps(self.apple):
                    break
            else:
                break

    def update(self, dt):
        self.steps += 1
        if self._external:
            # if an external control system is hooked up, call it with the game instance
            self._external(self)
        # forbid the head from running into the body
        if (self.current_direction - self.direction + 2) % 4 != 0:
            if self.current_direction != self.direction:
                self.turns += 1
            self.current_direction = self.direction
        # move the snake by popping it's last segment, and updating it's position so that it'll be where the head is
        last_segment = self.snake.pop()
        last_segment.position = self.head.position
        self.snake.insert(1, last_segment)
        # move the head itself
        if self.current_direction == 0:
            self.head.move(dx=20)
        elif self.current_direction == 1:
            self.head.move(dy=20)
        elif self.current_direction == 2:
            self.head.move(dx=-20)
        elif self.current_direction == 3:
            self.head.move(dy=-20)
        # detect apple eating / collision
        if self.head.overlaps(self.apple):
            self.score += 1
            self.steps = 0
            self.turns = 0
            self.snake.append(Sprite(snake_seg, 700, 700, batch=self.main_batch))
            self.label.text = f"Score: {self.score}"
            self.gen_apple()
        # detect collision with the snake body
        for segment in self.snake[1:]:
            if self.head.overlaps(segment):
                self.exit_code = 1
                self.event_loop.exit()
        # detect out-of-bounds movement
        if not self.head.overlaps(self.window, fully=True):
            self.exit_code = 2
            self.event_loop.exit()
        # detect stuck controller
        if self.steps > min(self.score * 100 + 200, 1000):
            self.exit_code = 3
            self.event_loop.exit()

    def reset(self):
        # Please check that things here have been updated to match things from the init constructor
        self.snake = self.snake[:5]
        self.score = 0
        self.steps = 0
        self.turns = 0
        self.exit_code = 0
        self.direction = 0
        self.current_direction = 0
        self.label.text = "Score: 0"
        self.gen_apple()
        for segment, x in zip(self.snake, range(290, 190, -20)):
            segment.update(x, 290)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.RIGHT:
            self.direction = 0
        elif symbol == key.UP:
            self.direction = 1
        elif symbol == key.LEFT:
            self.direction = 2
        elif symbol == key.DOWN:
            self.direction = 3
        elif symbol == key.SPACE:
            self.fps = 20

    def on_key_release(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.fps = 10

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps: int):
        self.clock.unschedule(self.update)
        if fps > 0:
            if fps > 150:  # faster than the refresh rate, so just schedule for every frame
                self.clock.schedule(self.update)
            else:
                self.clock.schedule_interval(self.update, 1 / fps)

    @property
    def external(self):
        return self._external

    @external.setter
    def external(self, func: Optional[Callable]):
        if func is None:
            self.window.event(self.on_key_press)
            self.window.event(self.on_key_release)
        else:
            self.window.remove_handler("on_key_press", self.on_key_press)
            self.window.remove_handler("on_key_release", self.on_key_release)
        self._external = func


if __name__ == "__main__":
    game = Game()
    while not game.window.has_exit:
        game.run()
        print(game.score)
        if game.window.has_exit:
            break
        sleep(2)
        game.reset()