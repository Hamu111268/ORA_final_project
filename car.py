from priority_queue import PriorityQueue

class Car:
    # Seats of the car
    capacity = 2

    def __init__(self, position, grid_map):
        self.position = position
        self.grid_map = grid_map

        # Possible status:
        # 1. idle
        # 2. picking_up
        # 3. dropping_off
        self.status = "idle"

        self.serving = [] # Serving passengers
        self.pending = [] # Pending passengers
        self.path = []

    def __repr__(self):
        msg = ""
        msg += f"cls: {type(self).__name__}"
        msg += f", id: {hex(id(self))}"
        msg += f", status: {self.status}"
        msg += f", position: {str(self.position)}"
        msg += f", path: {str(self.path)}"

        return msg

    def assign_passenger(self, passenger):
        self.pending.append(passenger)

    def pair_passengers(self, duration):
        assert len(self.serving) == 0

        if len(self.pending) == 0: # No pending passenger to serve
            self.status = "idle"
            return

        self.serving.append(self.pending.pop(0))

        # Ride sharing
        for i in range(1, self.capacity):
            if len(self.pending) == 0:
                break

            Row_max, Col_max = self.serving[i - 1].pick_up_point
            Row_min, Col_min = self.serving[i - 1].drop_off_point

            if Row_max < Row_min:
                Row_max, Row_min = Row_min, Row_max
            if Col_max < Col_min:
                Col_max, Col_min = Col_min, Col_max

            for j in range(len(self.pending)):
                x, y = self.pending[j].pick_up_point
                if Row_min <= x <= Row_max and Col_min <= y <= Col_max:
                    self.serving.append(self.pending.pop(j))
                    break

        points = [self.position]
        for p in self.serving:
            if p.pick_up_point == self.position:
                p.status = "picked_up"
                p.waiting_steps = duration
            else:
                p.status = "wait_pick"
                points.append(p.pick_up_point)

        self.path = self.grid_map.plan_path(points)

        if len(self.path) > 0:
            self.status = "picking_up"
        else:
            self.plan_drop_off_path(duration)

    def plan_drop_off_path(self, duration):
        assert len(self.serving) > 0

        self.status = "dropping_off"

        cmp = lambda pa, pb: self.grid_map.dist_between(self.position, pa) < self.grid_map.dist_between(self.position, pb)
        pq = PriorityQueue(cmp)

        """
        Drop passengers according to the distance
        between current position and their destinations.
        """
        for i in range(len(self.serving) - 1, -1, -1): # reverse order
            p = self.serving[i]
            if p.drop_off_point == self.position:
                p.status = "dropped_off"
                self.serving.pop(i)
            else:
                pq.push(p.drop_off_point)

        points = [self.position]
        while len(pq) > 0:
            points.append(pq.top())
            pq.pop()

        self.path = self.grid_map.plan_path(points)

        if len(self.path) == 0:
            self.pair_passengers(duration)

    """
    Move the car to the
    next position on the path.
    """
    def move_to_next_position(self, duration):
        assert self.status != "idle", "Shouldn't move"

        self.position = self.path.pop(0)

        for i in range(len(self.serving) - 1, -1, -1): # reverse order
            passenger = self.serving[i]
            if self.status == "picking_up" and self.position == passenger.pick_up_point:
                passenger.status = "picked_up"
                passenger.waiting_steps = duration
            elif self.status == "dropping_off" and self.position == passenger.drop_off_point:
                passenger.status = "dropped_off"
                self.serving.pop(i)

        if len(self.path) == 0:
            if self.status == "picking_up":
                self.plan_drop_off_path(duration)
            else:
                self.pair_passengers(duration)


if __name__ == "__main__":
   pass
