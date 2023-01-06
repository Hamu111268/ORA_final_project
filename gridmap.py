import random
import pickle
import os
import shutil

from car import Car
from passenger import Passenger
from priority_queue import PriorityQueue

class GridMap:
    def __init__(self, rnd_seed, size, num_cars, num_passengers, num_drop = 0):
        self.rnd_seed = rnd_seed
        self.size = size # (row, col)
        self.num_cars = num_cars
        self.num_passengers = num_passengers

        self.cars = []
        self.passengers = []
        self.map_cost = dict()

        random.seed(rnd_seed)
        self.add_cars(self.num_cars)
        self.add_passengers(self.num_passengers)
        self.init_map_cost()
        self.random_drop_edges(num_drop)

        self.__save_dir = "./prev_dump/"
        if os.path.exists(self.__save_dir):
            shutil.rmtree(self.__save_dir)

        os.mkdir(self.__save_dir)

    """
    Test if `p` is a valid coordinate in grid map.
    """
    def is_valid(self, p):
        return (0 <= p[0] < self.size[0]) and (0 <= p[1] < self.size[1])

    """
    Test if `p1` is adjacent to `p2`.
    """
    def is_adjacent(self, p1, p2):
        assert self.is_valid(p1), "map point p1 is out of boundary"
        assert self.is_valid(p2), "map point p2 is out of boundary"

        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1

    """
    Set the weight of edges in the grid map with random value
    """
    def init_map_cost(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                p = (row, col)

                p_down = (row + 1, col)
                if self.is_valid(p_down):
                    self.map_cost[(p_down, p)] = self.map_cost[(p, p_down)] = random.randint(1, 9)

                p_right = (row, col + 1)
                if self.is_valid(p_right):
                    self.map_cost[(p_right, p)] = self.map_cost[(p, p_right)] = random.randint(1, 9)

    """
    Similar to init_map_cost(), but the weight is set to 1
    """
    def init_one_map_cost(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                p = (row, col)

                p_down = (row + 1, col)
                if self.is_valid(p_down):
                    self.map_cost[(p_down, p)] = self.map_cost[(p, p_down)] = 1

                p_right = (row, col + 1)
                if self.is_valid(p_right):
                    self.map_cost[(p_right, p)] = self.map_cost[(p, p_right)] = 1

    """
    Drop `num_drop` edges randomly
    """
    def random_drop_edges(self, num_drop):
        drop = random.sample(sorted(self.map_cost), num_drop)
        for key in drop:
            del self.map_cost[key]

    """
    Add `num_cars` cars with random state
    state: (current_x, current_y)
    """
    def add_cars(self, num_cars):
        assert num_cars <= self.size[0] * self.size[1], "number of cars is larger than number of grids"

        car_set = set()
        while len(car_set) < num_cars:
            p = (random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1))
            car_set.add(p)

        for s in car_set:
            self.cars.append(Car(s, self))

    """
    Add `num_passengers` passengers with random state
    state: (start_x, start_y, target_x, target_y)
    """
    def add_passengers(self, num_passengers):
        assert num_passengers <= self.size[0] * self.size[1], "number of passengers is larger than number of grids"

        passenger_set = set()
        while len(passenger_set) < num_passengers:
            p = (random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1))
            passenger_set.add(p)

        for sx, sy in passenger_set:
            # generate destination point
            dx = (sx + random.randint(1, self.size[0] - 1)) % self.size[0]
            dy = (sy + random.randint(1, self.size[1] - 1)) % self.size[1]

            self.passengers.append(Passenger((sx, sy), (dx, dy)))

    """
    Calcute shortest distance from `p1` to `p2`
    """
    def dist_between(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    """
    Find the shortest path from `start_point` to `end_point` by Dijkstra Algorithm.

    Returned path: (start_point, end_point]
    """
    def plan_path_two_points(self, start_point, end_point):
        prev = self.__get_prev(start_point)

        path = []

        cur = end_point
        while cur != start_point:
            path.append(cur)
            _, cur = prev[cur]

        path.reverse()

        return path

    """
    Find the shortest path going through every point in `points`.
    """
    def plan_path(self, points):
        path = []
        for i in range(1, len(points)):
            path += self.plan_path_two_points(points[i - 1], points[i])

        return path

    def __get_prev(self, start_point):
        pickle_file = os.path.join(self.__save_dir, str(start_point))
        if not os.path.exists(pickle_file):
            self.__single_source_shortest_path(start_point, pickle_file)

        with open(pickle_file, "rb") as f:
            prev = pickle.load(f)

        return prev

    def __single_source_shortest_path(self, start_point, save_file):
        prev = dict()

        # Element type is (time, (x, y), (prev_x, prev_y))
        pq = PriorityQueue(lambda lhs, rhs : lhs[0] < rhs[0])

        # Use Dijkstra Algorithm to solve
        # single source shortest path problem
        pq.push((0, start_point, None))
        while len(pq) > 0:
            t, (x, y), p = pq.top()
            pq.pop()

            # Out of date
            if (x, y) in prev:
                continue

            prev[(x, y)] = (t, p);

            for nxt_pos in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if ((x, y), nxt_pos) in self.map_cost:
                    pq.push((t + self.map_cost[(x, y), nxt_pos], nxt_pos, (x, y)))

        with open(save_file, "wb") as f:
            pickle.dump(prev, f)

if __name__ == "__main__":
    g = GridMap(1, (100, 100), 10, 20, 1000)

    print(g.plan_path_two_points((0, 0), (99, 99)))

    print(f"distance: {g.dist_between((0, 0), (99, 99))}")
