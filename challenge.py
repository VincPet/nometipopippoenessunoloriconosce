from concurrent.futures import thread
from heapq import heappush, heappop
from threading import Thread

import numpy as np
import os

from multiprocessing import freeze_support

try:
    import Queue as Queue  # ver. < 3.0
except ImportError:
    import queue as Queue


def load_file(file_name):
    file_object = open(file_name, "r").read().splitlines()
    first = file_object[0].split(" ")

    startpoint = np.array(first[:2])
    startpoint = startpoint.astype(int)
    endpoint = np.array(first[2:])
    endpoint = endpoint.astype(int)

    triang_num = int(file_object[1])
    triangles = np.empty([int(triang_num), 3, 2])

    file_object = file_object[2:]

    for idx, line in enumerate(file_object):
        triangles[idx] = parse_line(line)

    return startpoint, endpoint, triangles.astype(int), triang_num


def parse_line(line):
    points = np.array(line.split(" "))
    points = points.astype(int)

    return np.reshape(points, [3, 2])


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def PointInTriangle(pt, v1, v2, v3):
    b1 = sign(pt, v1, v2) < 0
    b2 = sign(pt, v2, v3) < 0
    b3 = sign(pt, v3, v1) < 0

    return (b1 == b2) and (b2 == b3)


def point_is_in_triangle(p):
    # return p in triangle_points
    result = checkedPoints.setdefault(p, None)
    if (result == None):
        result = False
        for triangle in triangles:
            isInTriangle = PointInTriangle(p, triangle[0], triangle[1], triangle[2])
            checkedPoints[p] = isInTriangle

            if (isInTriangle):
                return True

    return result


def is_in_space(point):
    return not (point[0] < min_x or point[0] > max_x or point[1] > max_y or point[1] < min_y)


def get_neighbour(point):
    result = []
    for i in range(-1, 2):
        for j in range(-1, 2):

            if i == 0 and j == 0:
                continue

            neighbour = getPoint((point[0] + i, point[1] + j))
            if is_in_space(neighbour) and not point_is_in_triangle(neighbour):
                result.append(neighbour)
    return result


def diff(v1, v2):
    return v1[0] - v2[0], v1[1] - v2[1]


def distance(p1, p2):
    return np.sqrt(np.power(p1[0] - p2[0], 2) + np.power(p1[1] - p2[1], 2))


def equals(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1]


def getPrev(prev, curr, nexts):
    prev_cur = diff(prev, curr)

    if len(nexts) == 1:
        return nexts[0]

    for next in nexts:
        curr_next = diff(curr, next)
        if (equals(prev_cur, curr_next)):
            return next

    return None


def get_bounding_box(a, b, c):
    coord = 0
    min_x = min(a[coord], min(b[coord], c[coord]))
    max_x = max(a[coord], max(b[coord], c[coord]))
    coord = 1
    min_y = min(a[coord], min(b[coord], c[coord]))
    max_y = max(a[coord], max(b[coord], c[coord]))

    return min_x, max_x, min_y, max_y


class PriorityQueue2:

    def __init__(self):
        self._queue = []
        self._index = 0
        self._entry_finder = {}  # mapping of tasks to entries
        self.pq = Queue.PriorityQueue()

    def push(self, item, priority, priority_2):
        p = getPoint(item)
        item = [True, p]

        if p in self._entry_finder:
            self._entry_finder[p][0] = False
            del self._entry_finder[p]
        self._entry_finder[p] = item

        obj = [(priority*1000 + priority_2), priority, self._index, item]
        # self.pq.put(obj)
        heappush(self._queue, obj)
        self._index += 1

    def pop(self):
        while self.hasElems():
            # priority, priority2, index, item = self.pq.get()
            priority, priority2, index, item = heappop(self._queue)

            if item[0]:
                del self._entry_finder[item[1]]
                return priority2, item[1]

        return -1, -1

    def hasElems(self):
        # return not self.pq.empty()
        return self._queue


triangle_points = set()
points = {}


def plotLineLow(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    D = 2 * dy - dx
    y = y0
    tmp = set()
    for x in range(x0, x1):
        # triangle_points.add(getPoint((x, y)))
        tmp.add(getPoint((x, y)))

        if D > 0:
            y = y + yi
            D = D - 2 * dx

        D = D + 2 * dy
    return tmp


def plotLineHigh(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2 * dx - dy
    x = x0

    tmp = set()
    for y in range(y0, y1):
        # triangle_points.add(getPoint((x, y)))
        tmp.add(getPoint((x, y)))
        if D > 0:
            x = x + xi
            D = D - 2 * dy
        D = D + 2 * dx
    return tmp


def triangle_points_pop(x0, y0, x1, y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return plotLineLow(x1, y1, x0, y0)
        else:
            return plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            return plotLineHigh(x1, y1, x0, y0)
        else:
            return plotLineHigh(x0, y0, x1, y1)


def add_triangle(a, b, c):
    return triangle_points_pop(a[0], a[1], b[0], b[1]) | \
           triangle_points_pop(b[0], b[1], c[0], c[1]) | \
           triangle_points_pop(a[0], a[1], c[0], c[1])


def getPoint(p):
    result = points.setdefault(p[0], {})
    return result.setdefault(p[1], (p[0], p[1]))


def Dijkstra(start, end):
    D = {}  # dictionary of final distances
    P = {}  # dictionary of predecessors

    tmp_dist = {}
    start = getPoint(start)
    end = getPoint(end)

    Q2 = PriorityQueue2()
    Q2.push(start, 0, 0)
    tmp_dist[start] = 0

    P[start] = [start]

    while Q2.hasElems():

        dist, v = Q2.pop()

        if (dist == -1):
            break

        print(v)
        D[v] = dist
        if v == end: break

        for w in get_neighbour(v):
            weight = 1

            for p in P[v]:
                dir_pv_v = diff(p, v)
                dir_v_w = diff(v, w)

                if (dir_pv_v[0] == dir_v_w[0] and dir_pv_v[1] == dir_v_w[1]) or (dir_pv_v == (0, 0)):
                    weight = 0
                    break

            vwEdges = dist + weight
            dist_w = tmp_dist.setdefault(w, -1)
            if w in D:
                continue
            if dist_w == -1 or vwEdges < dist_w:
                Q2.push(w, vwEdges, distance(w, end))
                tmp_dist[w] = vwEdges
                P[w] = [v]
            elif vwEdges == dist_w:
                P[w].append(v)
                tmp_dist[w] = vwEdges
                Q2.push(w, vwEdges, distance(w, end))

    return (D, P)


def get_best_node(candidates, prev, curr):
    prev_cur = diff(prev, curr)

    for candidate in candidates:
        cur_cand = diff(curr, candidate)
        if equals(prev_cur, cur_cand):
            return candidate

    return candidates[0]


def shortestPath(start, end=None):
    D, P = Dijkstra(start, end)

    Path = []
    Path.append(end)
    end = [end]

    while 1:
        if (len(end) == 1):
            end = end[0]
        else:
            prev = Path[-1]
            prev_prev = Path[-2]

            end = get_best_node(end, prev_prev, prev)

        Path.append(end)
        if end == start:
            break
        end = P[getPoint(end)]

    Path.reverse()
    return Path


def plot(path):
    from matplotlib import pyplot as plt
    # .....

    # ..... io avevo le coordinate del percorso in un dizionario chiamato path, quindi sono dovuto passare a np.array per comodità

    path2 = np.array(list(path)).reshape(-1, 2)
    pippo_x, pippo_y = path2.T
    plt.plot(pippo_x, pippo_y)

    for i in range(0, len(triangles)):
        triangle = triangles[i]
        t = plt.Polygon(triangle, color='red')
        plt.gca().add_patch(t)

    # aggiustati la scala delle x e y per visualizzare meglio tutto
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.show()


def add_triangles(start, end):
    print(start)
    print(end)
    tmp = set()
    for i in range(int(start), int(end)):
        t = triangles[i]
        tmp = tmp | add_triangle(t[0], t[1], t[2])

    triangle_points.update(tmp)


curdir = os.getcwd()
filename = os.path.join(curdir, '/input_1.txt')

start, end, triangles, num = load_file(filename)

if __name__ == '__main__':

    min_x = min(start[0], end[0])
    min_y = min(start[1], end[1])
    max_x = max(start[0], end[0])
    max_y = max(start[1], end[1])
    i = 0

    for triangle in triangles:
        i += 1
        for point in triangle:
            min_x = min(min_x, point[0])
            min_y = min(min_y, point[1])
            max_x = max(max_x, point[0])
            max_y = max(max_y, point[1])

    print("triang_start")

    n_th = 1
    for_th = int(num / n_th)
    threads = []

    # import multiprocessing as mp

    # pool = mp.Pool(processes=for_th)
    # results = [pool.apply_async(add_triangle, args=(x * for_th, (x + 1) * for_th)) for x in range(0, n_th)]

    # [triangle_points.update(p.get()) for p in results]
    checkedPoints = {}
    print("triang_end")
    start = (start[0], start[1])
    end = (end[0], end[1])
    path = shortestPath(start, end)
    print(path)

    plot(path)
