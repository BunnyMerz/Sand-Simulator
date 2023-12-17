from typing import Any
from numba import cuda
import numpy as np

from pygame import Surface, SRCALPHA, draw as py_draw

from screen import Screen

def matrix_to_number(matrix: list[tuple[int]]):
    return (
        matrix[1][2] * 3**0 +

        matrix[0][2] * 3**1 +
        matrix[0][1] * 3**2 +
        matrix[0][0] * 3**3 +

        matrix[1][0] * 3**4 +

        matrix[2][0] * 3**5 +
        matrix[2][1] * 3**6 +
        matrix[2][2] * 3**7
    )
def all_possibilities(matrix: list[tuple[int]]):
    for y in range(3):
        for x in range(3):
            if matrix[y][x] == 2:
                r = []
                matrix[y][x] = 0
                r += all_possibilities(matrix)
                matrix[y][x] = 1
                r += all_possibilities(matrix)
                matrix[y][x] = 2
                return r
    m = []
    for l in matrix:
        m.append([c for c in l])
    return [m]

WIDTH = 100
HEIGHT = 100
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

    particle_rules_const = np.zeros(shape=(3**8, 2), dtype=np.int8)
    simple_rules = [
        # Sand/Water/Gase
        ([[2, 2, 2],
         [2, 0, 2],
         [2, 0, 2],], (0,-1)),
         
        ([[2, 2, 2],
         [2, 0, 2],
         [2, 1, 0],], (1,-1)),

        ([[2, 2, 2],
         [2, 0, 2],
         [0, 1, 1],], (-1,-1)),

         # Water/Gas
        ([[2, 2, 2],
         [1, 0, 0],
         [1, 1, 1],], (1,0)),

         ([[2, 2, 2],
         [0, 0, 1],
         [1, 1, 1],], (-1,0)),
    ]
    for rule, (x,y) in simple_rules:
        for possibility in all_possibilities(rule):
            value = matrix_to_number(possibility)
            particle_rules_const[value][0] = x
            particle_rules_const[value][1] = y

    particle_rules_const = cuda.to_device(particle_rules_const)

    def __init__(self, w, h, chunk_width) -> None:
        self.max_particles = w*h

        self.chunck_width = chunk_width
        self.chunck_heigth = h
        self.grid_amount = (w // self.chunck_width) + 1

        self.grid = np.zeros(shape=(self.grid_amount), dtype=np.int8) # List of where each grid starts [0, 2, 10, 30] => index at x/y
        self.active = np.zeros(shape=(self.max_particles), dtype=np.bool_)

        self.d_board = cuda.to_device(np.zeros(shape=(self.grid_amount, self.chunck_heigth, self.chunck_width), dtype=np.int8))

        self.ptype = np.zeros(shape=(self.max_particles), dtype=np.int8)
        self.x = np.zeros(shape=(self.max_particles), dtype=np.int32)
        self.y = np.zeros(shape=(self.max_particles), dtype=np.int32)


        self.threadsperblock = 256
        self.blockspergrid = (self.max_particles + (self.threadsperblock - 1)) // self.threadsperblock

    def get_particle(self, i):
        if self.active[i] == False:
            return None
        return (self.ptype[i], self.x[i], self.y[i])
    def set_particle(self, i, x, y):
        self.active[i] = True
        self.x[i] = x
        self.y[i] = y

    def render(self, width, height):
        print("render start")
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

        print("render end")
        return output_surf
    
    def from_device(self, d_grid, d_active, d_x, d_y):
        # self.grid = d_grid.copy_to_host()
        # self.active = d_active.copy_to_host()
        self.x = d_x.copy_to_host()
        self.y = d_y.copy_to_host()
    def to_device(self):
        return (
            cuda.to_device(self.grid),
            self.d_board,
            cuda.to_device(self.active),
            cuda.to_device(self.ptype),
            cuda.to_device(self.x),
            cuda.to_device(self.y)
        )

    @staticmethod
    @cuda.jit
    def update_particles(grid: list[int], board: tuple[tuple[tuple[int]]], x: list[int], y: list[int], grid_size: int, par_size: int, board_size: int):
        i = cuda.grid(1)
        if i < board_size:
            board_i = board[i]
            start_i = grid[i]
            end_i = 0
            # Particles range: [start_i, end_i[
            if start_i != None: # There are particles in this grid
                look_x = i + 1
                while(look_x < grid_size and grid[look_x] == None):
                    look_x += 1
                end_i = par_size
                if look_x < grid_size:
                    end_i = grid[look_x]

                par_i = start_i
                while(par_i < end_i):
                    # Bool analysis
                    x[par_i] += 1
                    par_i += 1


    def update(self):
        print("======")
        print("to device")
        d_grid, d_board, d_active, d_ptype, d_x, d_y = self.to_device()
        cuda.synchronize()
        print("update")
        self.update_particles[self.blockspergrid, self.threadsperblock](d_grid, d_board, d_x, d_y, self.grid_amount, self.max_particles, self.grid_amount)
        print("updated")
        cuda.synchronize()
        self.from_device(d_grid, d_active, d_x, d_y)
        print("from device")
        print("==")

def main():
    ws = Screen(WIDTH, HEIGHT)

    pars = []
    for part_type in range(1):
        ps = Particles(WIDTH, HEIGHT, 10)
        for x in range(WIDTH * HEIGHT):
            ps.set_particle(x, WIDTH_IN_PARTICLES//2 + x % 10, x//10 * 2)
        pars.append(ps)

    ws.update([x.render(ws.width, ws.height) for x in pars])
    ws.clock.tick(60)
    ws.update([x.render(ws.width, ws.height) for x in pars])

    while(1):
        ws.clock.tick(60)
        print("cycle")
        [x.update() for x in pars]
        ws.update([x.render(ws.width, ws.height) for x in pars])
        print("rendered")


if __name__ == "__main__":
    main()