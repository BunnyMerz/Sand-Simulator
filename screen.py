import pygame
from pygame import display, Surface
from pygame.time import Clock
import sys


class Screen():
    """Main class to hangle window and other pygame responsabilites"""
    def __init__(self, height: int, width: int) -> None:
        pygame.init()
        pygame.key.set_repeat(200, 25)

        self.clock = Clock()
        self.window: pygame.Surface = display.set_mode((height,width),pygame.RESIZABLE)
        display.set_caption("Now with Falling sand!")
        display.set_icon(pygame.image.load("logo.png"))

    def terminate(self):
        pygame.quit()
        sys.exit(0)

    @property
    def width(self):
        return self.window.get_width()
    @property
    def height(self):
        return self.window.get_height()
    
    def update(self, to_draw: list[Surface] = None, x=0,y=0):
        self.window.fill((30,30,30))
        # delta = self.clock.tick(120) / 1000
        events = pygame.event.get()
        for e in events:
            if pygame.QUIT == e.type: ## Avoid freezing
                self.terminate()
        keys = pygame.key.get_pressed()

        # e.draw(self.window, self.width/2, self.height/2)
        if to_draw is not None:
            for s in to_draw:
                self.window.blit(s, (x,y))
        display.update()