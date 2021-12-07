"""
3DRenderer - A basic 3D rendering/rasterization engine capable of unwrapping .obj files, with exclusively triangles,
instantiating, light sources, and real-time transformations and manipulations of objects
"""

import math
import pygame as pg
import random

WIN = pg.display.set_mode((0, 0), pg.FULLSCREEN, 64)
WIN_WIDTH = pg.display.get_window_size()[0]
WIN_HEIGHT = pg.display.get_window_size()[1]

pg.init()
pg.display.set_caption("3D Rendering")

WIREFRAME = True


def sort_triangles(tria):
    return -tria.avg_z()

def multiplymatrixvector(vector, matrix):
    x = vector[0] * matrix[0][0] + vector[1] * matrix[1][0] + vector[2] * matrix[2][0] + matrix[3][0]
    y = vector[0] * matrix[0][1] + vector[1] * matrix[1][1] + vector[2] * matrix[2][1] + matrix[3][1]
    z = vector[0] * matrix[0][2] + vector[1] * matrix[1][2] + vector[2] * matrix[2][2] + matrix[3][2]
    w = vector[0] * matrix[0][3] + vector[1] * matrix[1][3] + vector[2] * matrix[2][3] + matrix[3][3]

    if not w == 0.0:
        x /= w
        y /= w
        z /= w
    return [x, y, z]


def normalize(vecta):
    vectb = [0]*len(vecta)
    div = 0.0
    for i in range(len(vecta)):
        div += vecta[i] * vecta[i]
    div = math.sqrt(div)
    for i in range(len(vecta)):
        vectb[i] = vecta[i] / div
    return vectb


def dot(vecta, vectb):
    return_dot = 0.0
    for i in range(len(vecta)):
        return_dot += vecta[i] * vectb[i]
    return return_dot


def subtract(vecta, vectb):
    return_vect = [0]*len(vecta)
    for i in range(len(vecta)):
        return_vect[i] = vecta[i] - vectb[i]
    return return_vect


class Triangle:
    def __init__(self, vertices):
        self.vertices = vertices

    def set_vertices(self, other_triangle):
        for i in range(len(self.vertices)):
            for j in range(len(self.vertices[i])):
                self.vertices[i][j] = other_triangle.vertices[i][j]

    def normal(self):
        l1 = [self.vertices[1][0] - self.vertices[0][0], self.vertices[1][1] - self.vertices[0][1],
              self.vertices[1][2] - self.vertices[0][2]]
        l2 = [self.vertices[2][0] - self.vertices[0][0], self.vertices[2][1] - self.vertices[0][1],
              self.vertices[2][2] - self.vertices[0][2]]
        normal = [l1[1] * l2[2] - l1[2] * l2[1], l1[2] * l2[0] - l1[0] * l2[2], l1[0] * l2[1] - l1[1] * l2[0]]
        return normal

    def avg_z(self):
        avg = 0.0
        for i in range(3):
            avg += self.vertices[i][2]
        return avg / 3.0


class Mesh:
    def __init__(self, triangles):
        self.triangles = []*len(triangles)
        for i in range(len(triangles)):
            self.triangles.append(Triangle(triangles[i]))


class Camera:
    def __init__(self, cam_pos, cam_dir, frustum_angle):
        self.pos = cam_pos
        self.dir = cam_dir
        self.fov = frustum_angle


class Rendering_Engine:
    cube = Mesh([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                 [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                 [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                 [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                 [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                 [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
                 [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                 [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                 [[0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                 [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                 [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                 [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]])

    def __init__(self):
        self.cube = Mesh(self.load_from_file('files/amogus.obj'))

        self.camera = Camera([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], math.pi / 4)
        self.zfar = 1000.0
        self.znear = 0.1
        self.a = WIN_HEIGHT / WIN_WIDTH
        self.f = 1.0 / math.tan(self.camera.fov / 2.0)
        self.q = self.zfar / (self.zfar - self.znear)
        self.projection_matrix = [[self.a*self.f, 0.0, 0.0, 0.0],
                                  [0.0, self.f, 0.0, 0.0],
                                  [0.0, 0.0, self.q, 1.0],
                                  [0.0, 0.0, -self.znear * self.q, 0.0]]

    def draw(self):
        pg.draw.rect(WIN, (255, 255, 255), (0, WIN_HEIGHT/2, WIN_WIDTH, WIN_HEIGHT))
        drawn_triangles = list()
        for i in range(len(self.cube.triangles)):

            translated_triangle = Triangle([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            translated_triangle.set_vertices(self.cube.triangles[i])

            fTheta = pg.time.get_ticks() / 10000

            """translated_triangle.vertices[0][0] -= 0.5
            translated_triangle.vertices[1][0] -= 0.5
            translated_triangle.vertices[2][0] -= 0.5
            translated_triangle.vertices[0][1] -= 0.5
            translated_triangle.vertices[1][1] -= 0.5
            translated_triangle.vertices[2][1] -= 0.5
            translated_triangle.vertices[0][2] -= 0.5
            translated_triangle.vertices[1][2] -= 0.5
            translated_triangle.vertices[2][2] -= 0.5"""

            translated_triangle = self.rotate_triangle_z(translated_triangle, fTheta)
            translated_triangle = self.rotate_triangle_x(translated_triangle, fTheta*3.0)

            translated_triangle.vertices[0][2] += 500.0# + math.cos(fTheta)
            translated_triangle.vertices[1][2] += 500.0# + math.cos(fTheta)
            translated_triangle.vertices[2][2] += 500.0# + math.cos(fTheta)

            point = translated_triangle.vertices[0]
            if dot(translated_triangle.normal(), subtract(point, self.camera.pos)) < 0:
                drawn_triangles.append(translated_triangle)
        drawn_triangles.sort(key=sort_triangles)

        for translated_triangle in drawn_triangles:
            v1 = multiplymatrixvector(translated_triangle.vertices[0], self.projection_matrix)
            v2 = multiplymatrixvector(translated_triangle.vertices[1], self.projection_matrix)
            v3 = multiplymatrixvector(translated_triangle.vertices[2], self.projection_matrix)

            v1[0] += 1.0
            v1[0] *= 0.5 * WIN_WIDTH
            v1[1] += 1.0
            v1[1] *= 0.5 * WIN_HEIGHT

            v2[0] += 1.0
            v2[0] *= 0.5 * WIN_WIDTH
            v2[1] += 1.0
            v2[1] *= 0.5 * WIN_HEIGHT

            v3[0] += 1.0
            v3[0] *= 0.5 * WIN_WIDTH
            v3[1] += 1.0
            v3[1] *= 0.5 * WIN_HEIGHT

            vect1 = normalize(translated_triangle.normal())
            li = (dot(vect1, normalize([0, -1, 0])) + 1) / 2.0
            pg.draw.polygon(WIN, (li*173, li*216, li*230), ((v1[0], WIN_HEIGHT - v1[1]), (v2[0], WIN_HEIGHT - v2[1]), (v3[0], WIN_HEIGHT - v3[1])))
            if WIREFRAME:
                pg.draw.polygon(WIN, (200, 100, 200), ((v1[0], WIN_HEIGHT - v1[1]), (v2[0], WIN_HEIGHT - v2[1]), (v3[0], WIN_HEIGHT - v3[1])), 3)


    def rotate_triangle_z(self, triangle, fTheta):
        rotz_matrix = [[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]]

        rotz_matrix[0][0] = math.cos(fTheta)
        rotz_matrix[0][1] = math.sin(fTheta)
        rotz_matrix[1][0] = -math.sin(fTheta)
        rotz_matrix[1][1] = math.cos(fTheta)
        rotz_matrix[2][2] = 1
        rotz_matrix[3][3] = 1

        for t in range(len(triangle.vertices)):
            triangle.vertices[t] = multiplymatrixvector(triangle.vertices[t], rotz_matrix)
        return triangle

    def rotate_triangle_x(self, triangle, fTheta):
        rotx_matrix = [[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]]

        rotx_matrix[0][0] = 1
        rotx_matrix[1][1] = math.cos(fTheta)
        rotx_matrix[1][2] = math.sin(fTheta)
        rotx_matrix[2][1] = -math.sin(fTheta)
        rotx_matrix[2][2] = math.cos(fTheta)
        rotx_matrix[3][3] = 1

        for t in range(len(triangle.vertices)):
            triangle.vertices[t] = multiplymatrixvector(triangle.vertices[t], rotx_matrix)
        return triangle

    def reinitialize_perspective(self):
        self.a = WIN_WIDTH / WIN_HEIGHT
        self.f = 1.0 / math.tan(self.camera.fov / 2.0)
        self.q = self.zfar / (self.zfar - self.znear)
        self.projection_matrix = [[self.a * self.f, 0.0, 0.0, 0.0],
                                  [0.0, self.f, 0.0, 0.0],
                                  [0.0, 0.0, self.q, 1.0],
                                  [0.0, 0.0, -self.znear * self.q, 0.0]]

    def load_from_file(self, filename):
        vertices = []
        triangles = []
        f = open(filename, 'r')
        for line in f:
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)

                vertex = [float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1])]
                vertex = [round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2)]

                vertices.append(vertex)
            elif line[0] == "f":
                i = line.find(" ") + 1
                triangle = []
                for item in range(3):
                    if line.find(" ", i) == -1:
                        triangle.append(vertices[int(line[i:-1]) - 1])
                        break
                    triangle.append(vertices[int(line[i:line.find(" ", i)]) - 1])
                    i = line.find(" ", i) + 1
                triangles.append(triangle)
        print(vertices)
        return triangles


if __name__ == "__main__":
    ge = Rendering_Engine()

    over = False
    while not over:
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    over = True
                if event.key == pg.K_w:
                    WIREFRAME = not WIREFRAME
        WIN.fill((0, 0, 0))
        ge.draw()
        pg.display.update()