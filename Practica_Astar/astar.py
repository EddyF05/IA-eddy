import pygame
import heapq
import math

pygame.init()

ANCHO_VENTANA = 800
TAM_GRID = 500
OFFSET = (ANCHO_VENTANA - TAM_GRID) // 2
EPSILON = 1.2

VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("A* Euclidiano Ponderado")

BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
GRIS = (128, 128, 128)
AZUL = (0, 0, 255)
VERDE = (0, 255, 0)
NARANJA = (255, 165, 0)
PURPURA = (128, 0, 128)

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = OFFSET + fila * ancho
        self.y = OFFSET + col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_abierto(self):
        self.color = AZUL

    def hacer_cerrado(self):
        self.color = VERDE

    def hacer_camino(self):
        self.color = NARANJA

    def vecinos(self, grid):
        vecinos = []
        movimientos = [
            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, 1.414), (1, -1, 1.414),
            (-1, 1, 1.414), (-1, -1, 1.414)
        ]

        for dx, dy, costo in movimientos:
            x = self.fila + dx
            y = self.col + dy
            if 0 <= x < self.total_filas and 0 <= y < self.total_filas:
                if not grid[x][y].es_pared():
                    vecinos.append((grid[x][y], costo))
        return vecinos

    def dibujar(self, ventana):
        pygame.draw.rect(
            ventana, self.color,
            (self.x, self.y, self.ancho, self.ancho)
        )

def heuristica(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def reconstruir_camino(came_from, actual, dibujar):
    while actual in came_from:
        actual = came_from[actual]
        if actual.color not in (PURPURA, NARANJA):
            actual.hacer_camino()
        dibujar()

def algoritmo_a_star(dibujar, grid, inicio, fin):
    contador = 0
    open_set = []
    heapq.heappush(open_set, (0, contador, inicio))

    came_from = {}
    g_score = {n: float("inf") for fila in grid for n in fila}
    g_score[inicio] = 0

    f_score = {n: float("inf") for fila in grid for n in fila}
    f_score[inicio] = EPSILON * heuristica(inicio.get_pos(), fin.get_pos())

    open_hash = {inicio}
    closed_set = set()

    while open_set:
        actual = heapq.heappop(open_set)[2]
        open_hash.discard(actual)

        if actual == fin:
            reconstruir_camino(came_from, fin, dibujar)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        closed_set.add(actual)

        if actual != inicio:
            actual.hacer_cerrado()

        for vecino, costo in actual.vecinos(grid):
            if vecino in closed_set:
                continue

            temp_g = g_score[actual] + costo
            if temp_g < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = temp_g
                f_score[vecino] = temp_g + EPSILON * heuristica(
                    vecino.get_pos(), fin.get_pos()
                )

                if vecino not in open_hash:
                    contador += 1
                    heapq.heappush(open_set, (f_score[vecino], contador, vecino))
                    open_hash.add(vecino)
                    if vecino != fin:
                        vecino.hacer_abierto()

        dibujar()

    return False

def crear_grid(filas):
    grid = []
    ancho_nodo = TAM_GRID // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            grid[i].append(Nodo(i, j, ancho_nodo, filas))
    return grid

def dibujar_grid(ventana, filas):
    ancho_nodo = TAM_GRID // filas
    for i in range(filas + 1):
        pygame.draw.line(
            ventana, GRIS,
            (OFFSET, OFFSET + i * ancho_nodo),
            (OFFSET + TAM_GRID, OFFSET + i * ancho_nodo)
        )
        pygame.draw.line(
            ventana, GRIS,
            (OFFSET + i * ancho_nodo, OFFSET),
            (OFFSET + i * ancho_nodo, OFFSET + TAM_GRID)
        )

def dibujar(ventana, grid, filas):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas)
    pygame.display.update()

def obtener_click_pos(pos, filas):
    x, y = pos
    if x < OFFSET or y < OFFSET or x > OFFSET + TAM_GRID or y > OFFSET + TAM_GRID:
        return None
    ancho_nodo = TAM_GRID // filas
    return (x - OFFSET) // ancho_nodo, (y - OFFSET) // ancho_nodo

def main():
    FILAS = 11
    grid = crear_grid(FILAS)
    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(VENTANA, grid, FILAS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:
                pos = obtener_click_pos(pygame.mouse.get_pos(), FILAS)
                if pos:
                    fila, col = pos
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != inicio and nodo != fin:
                        nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:
                pos = obtener_click_pos(pygame.mouse.get_pos(), FILAS)
                if pos:
                    fila, col = pos
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    algoritmo_a_star(
                        lambda: dibujar(VENTANA, grid, FILAS),
                        grid, inicio, fin
                    )

    pygame.quit()

main()
