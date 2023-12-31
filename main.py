import sys
from typing import Any
from numba import cuda
import numpy as np

from pygame import Surface, SRCALPHA, draw as py_draw

from screen import Screen


WIDTH = 800
HEIGHT = 400
PARTICLE_SIZE = 5
HEIGHT_IN_PARTICLES = HEIGHT//PARTICLE_SIZE - 1
WIDTH_IN_PARTICLES = WIDTH//PARTICLE_SIZE - 1

class Particles:
    pixel_size = 5
    colors = [
        (216, 158, 121),
        (40,40,90),
        (10,120,3),
    ]
    SAND = 0
    WATER = 1
    GAS = 2
    def __init__(self, max_particles) -> None:
        self.max_particles = max_particles
        self.active = np.zeros(shape=(max_particles), dtype=np.bool_)
        self.x = np.zeros(shape=(max_particles), dtype=np.int32)
        self.y = np.zeros(shape=(max_particles), dtype=np.int32)
        self.ptype = np.zeros(shape=(max_particles), dtype=np.int8)

        self.threadsperblock = 32
        self.blockspergrid = (max_particles + (self.threadsperblock - 1)) // self.threadsperblock

    def get_particle(self, i):
        if self.active[i] == False:
            return None
        return (self.ptype[i], self.x[i], self.y[i])
    def set_particle(self, i, x, y, ptype):
        self.active[i] = True
        self.x[i] = x
        self.y[i] = y
        self.ptype[i] = ptype

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
    def move_particles_sand(active, x, y, size):
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
    
    @staticmethod
    @cuda.jit
    def move_particles_gas(active, x, y, size):
        i = cuda.grid(1)
        move = True
        if i < size and active[i] == True and y[i] < HEIGHT_IN_PARTICLES:
            for other in range(0, size):
                if active[other] == True and y[other] == y[i] + 1 and x[other] == x[i]:
                    move = False
            if move == True:
                y[i] += 1
            else:
                move = True
                for other in range(0, size):
                    if active[other] == True and y[other] == y[i] + 1 and x[other] == x[i] - 1:
                        move = False
                if move == True:
                    x[i] -= 1
                    y[i] += 1
                else:
                    move = True
                    for other in range(0, size):
                        if active[other] == True and y[other] == y[i] + 1 and x[other] == x[i] + 1:
                            move = False
                    if move == True:
                        x[i] += 1
                        y[i] += 1
                    else:
                        left_available = True
                        right_available = True
                        for other in range(0, size):
                            if active[other] == True and y[other] == y[i] and x[other] == x[i] + 1:
                                right_available = False
                            elif active[other] == True and y[other] == y[i] and x[other] == x[i] - 1:
                                left_available = False

                        if right_available and not left_available:
                            x[i] += 1
                        elif not right_available and left_available:
                            x[i] -= 1

    @staticmethod
    @cuda.jit
    def move_particles_water(active, x, y, size):
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
                    else:
                        left_available = True
                        right_available = True
                        for other in range(0, size):
                            if active[other] == True and y[other] == y[i] and x[other] == x[i] + 1:
                                right_available = False
                            elif active[other] == True and y[other] == y[i] and x[other] == x[i] - 1:
                                left_available = False

                        if right_available and not left_available:
                            x[i] += 1
                        elif not right_available and left_available:
                            x[i] -= 1

    @staticmethod
    def CPU_update(i, active, ptype, x, y, size):
        if active[i] == True and y[i] > 0:
            move = True
            f11 = 0
            f12 = 0
            f13 = 0
            f21 = 0
            
            f23 = 0
            f31 = 0
            f32 = 0
            f33 = 0
            k = -1 if ptype[i] in [Particles.GAS] else -1
            for other in range(0, size):
                if active[other] == True:
                    if y[other] == y[i] + 1 and x[other] == x[i] - 1:
                        f11 = 1
                    if y[other] == y[i] + 1 and x[other] == x[i]:
                        f12 = 1
                    if y[other] == y[i] + 1 and x[other] == x[i] + 1:
                        f13 = 1
                    if y[other] == y[i]  and x[other] == x[i] - 1:
                        f21 = 1
                    if y[other] == y[i]  and x[other] == x[i] + 1:
                        f23 = 1
                    if y[other] == y[i] - 1 and x[other] == x[i] - 1:
                        f31 = 1
                    if y[other] == y[i] - 1 and x[other] == x[i]:
                        f32 = 1
                    if y[other] == y[i] - 1 and x[other] == x[i] + 1:
                        f33 = 1

            if f32 == 0:
                y[i] -= 1 * k
            elif f33 == 0:
                y[i] -= 1 * k
                x[i] += 1
            elif f31 == 0:
                y[i] -= 1 * k
                x[i] -= 1
            elif ptype[i] > 1:
                pass
            


    def update(self):
        if GPU:
            d_active, d_x, d_y = self.to_device()
            if self.ptype[0] == self.SAND:
                self.move_particles_sand[self.blockspergrid, self.threadsperblock](d_active, d_x, d_y, self.max_particles)
            if self.ptype[0] == self.WATER:
                self.move_particles_water[self.blockspergrid, self.threadsperblock](d_active, d_x, d_y, self.max_particles)
            if self.ptype[0] == self.GAS:
                self.move_particles_gas[self.blockspergrid, self.threadsperblock](d_active, d_x, d_y, self.max_particles)
            self.from_device(d_active, d_x, d_y)
        else:
            x = 0
            while(x < self.max_particles):
                self.CPU_update(
                    x,
                    self.active, self.ptype, self.x, self.y, self.max_particles
                )
                x += 1

GPU = "--cpu" not in sys.argv

def main():
    kp = 500
    ws = Screen(WIDTH, HEIGHT)

    pars = []
    for part_type in range(3):
        ps = Particles(kp)
        for x in range(kp):
            ps.set_particle(x, WIDTH_IN_PARTICLES//2 + x % 10, x//10 * 2, part_type)
        pars.append(ps)

    ws.update([x.render(ws.width, ws.height) for x in pars])
    ws.clock.tick(60)
    ws.update([x.render(ws.width, ws.height) for x in pars])

    x = 0
    while(x < 320):
        # ws.clock.tick(60)
        
        [x.update() for x in pars]
        ws.update([x.render(ws.width, ws.height) for x in pars])
        x += 1

    while(1): ws.update([x.render(ws.width, ws.height) for x in pars])


if __name__ == "__main__":
    main()