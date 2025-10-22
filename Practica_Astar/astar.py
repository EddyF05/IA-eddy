# astar.py
from heapq import heappush, heappop

def heuristic(a, b):
    # Manhattan
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, g, current, parent = heappop(open_set)
        if current in came_from:
            continue
        came_from[current] = parent

        if current == goal:
            # reconstruir camino
            path = []
            node = current
            while node:
                path.append(node)
                node = came_from[node]
            return list(reversed(path))

        x,y = current
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                tentative_g = g + 1
                neighbor = (nx,ny)
                if tentative_g < g_score.get(neighbor, 1e9):
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heappush(open_set, (f, tentative_g, neighbor, current))
    return None

if __name__ == "__main__":
    grid = [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,1,0],
        [0,1,0,0,0],
        [0,0,0,1,0],
    ]
    start = (0,0)
    goal = (4,4)
    path = astar(grid, start, goal)
    print("Path:", path)
