from typing import Any
from numba import cuda
import numpy as np

from pygame import Surface, SRCALPHA, draw as py_draw

from screen import Screen


class Particles:
    pixel_size = 5
    colors = [
        (40,40,40)
    ]
    def __init__(self, max_particles) -> None:
        self.max_particles = max_particles
        self.active = np.zeros(shape=(max_particles), dtype=np.bool_)
        self.x = np.zeros(shape=(max_particles), dtype=np.int32)
        self.y = np.zeros(shape=(max_particles), dtype=np.int32)

    def get_particle(self, i):
        if self.active[i] == False:
            return None
        return (0, self.x[i], self.y[i])
    def set_particle(self, i, x, y):
        self.active[i] = True
        self.x[i] = x
        self.y[i] = y

    def render(self, width, height):
        output_surf = Surface((width, height), SRCALPHA)

        for k in range(0, self.max_particles):
            part = self.get_particle(k)
            if part is not None:
                t, x, y = part
                py_draw.rect(
                    output_surf, self.colors[t],
                    (
                        x * self.pixel_size, -(y + 1) * self.pixel_size + height,
                        self.pixel_size, self.pixel_size
                    )
                )

        return output_surf
    
    def from_device(self, d_active, d_x, d_y):
        self.active = d_active.copy_to_host()
        self.x = d_x.copy_to_host()
        self.y = d_y.copy_to_host()
    def to_device(self):
        return cuda.to_device(self.active), cuda.to_device(self.x), cuda.to_device(self.y)

    # Naive solution for it
    @staticmethod
    @cuda.jit
    def move_particles(active, x, y, size):
        i = cuda.grid(1)
        move = True
        if active[i] == True and y[i] > 0:
            for other in range(0, size):
                if active[other] == True and y[other] == y[i] - 1 and x[other] == x[i]:
                    move = False
            if move == True:
                y[i] -= 1
            else:
                move = True
                for other in range(0, size):
                    if active[other] == True and y[other] == y[i] - 1 and x[other] == x[i] - 1:
                        move = False
                if move == True:
                    x[i] -= 1
                    y[i] -= 1
                else:
                    move = True
                    for other in range(0, size):
                        if active[other] == True and y[other] == y[i] - 1 and x[other] == x[i] + 1:
                            move = False
                    if move == True:
                        x[i] += 1
                        y[i] -= 1

    def update(self):
        d_active, d_x, d_y = self.to_device()
        self.move_particles[1, self.max_particles](d_active, d_x, d_y, self.max_particles)
        self.from_device(d_active, d_x, d_y)

def main():
    kp = 90
    ws = Screen(400, 400)
    pars = Particles(kp)
    ws.update(pars.render(ws.width, ws.height))

    for x in range(kp):
        pars.set_particle(x, 30, x*2+10)

    ws.update(pars.render(ws.width, ws.height))
    ws.clock.tick(60)
    ws.update(pars.render(ws.width, ws.height))

    while(1):
        ws.clock.tick(60)
        
        pars.update()
        ws.update(pars.render(ws.width, ws.height))


if __name__ == "__main__":
    main()