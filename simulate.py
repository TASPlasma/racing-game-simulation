import numpy as np
import math
import random
import matplotlib.pyplot as plt
from typing import List
from math_types import Vector2f, Matrix22f

def update_pos(x0, y0, v_x0, v_y0, i, dir='right'):
    """
    x0: initial horizontal position value
    y0: initial vertical position value
    v_x0: initial horizontal velocity component
    v_y0: initial vertical velocity component
    i: drift value, an integer between -7 and 7
    """

    v = update_vel(v_x0, v_y0, i, dir)
    v_x1, v_y1 = v[0], v[1]
    x1 = v_x1 + x0
    y1 = v_y1 + y0

    return np.array([x1, y1])

def update_vel(v_x0, v_y0, i, dir='right'):
    angle_dict = {
        -7: 0.003,
        -6: 0.004,
        -5: 0.005,
        -4: 0.006,
        -3: 0.007,
        -2: 0.008,
        -1: 0.009,
        0: 0.01,
        1: 0.011,
        2: 0.012,
        3: 0.013,
        4: 0.014,
        5: 0.015,
        6: 0.016,
        7: 0.017
    }
    theta = angle_dict[i] if (dir == 'right') else -1 * angle_dict[-1*i]
    # my_mat = RotateMatf(theta)
    my_mat = Matrix22f().rot_mat(theta)
    v_x1 = math.cos(theta) * v_x0 + math.sin(theta) * v_y0
    v_y1 = -1 * math.sin(theta) * v_x0 + math.cos(theta) * v_y0
    return np.array([v_x1, v_y1])


def mt_charge(inputs, dir = 'right'):
    """
    inputs: an array of drift values
    """
    charge = 0
    for input in inputs:
        if dir == 'left':
            if input <= -3:
                charge += 5
            else:
                charge += 2
        else:
            if input >= 3:
                charge += 5
            else:
                charge += 2
    return charge

def load_track(filename="track.txt"):
    try:
        outer_polygon = []
        inner_polygon = []
        
        with open(filename, "r") as file:
            current_polygon = None
            
            for line in file:
                line = line.strip()
                
                if line.startswith("Inner Curve"):
                    current_polygon = inner_polygon
                elif line.startswith("Outer Curve"):
                    current_polygon = outer_polygon
                elif line:
                    x, y = map(float, line.split(","))
                    current_polygon += [(x, y)]

            return outer_polygon, inner_polygon
    except (FileNotFoundError, ValueError):
        print(f"Error loading track data from {filename}. Returning empty polygons.")
        return [], []
    
def is_point_inside_polygon(x, y, polygon, tolerance=1e-10):
    """
    (x, y): a point to test if it is inside the polygon
    polygon: a list of vertices/points
    Returns whether the given point (x, y) is inside the polygon
    using the raycast method
    """
    intersections = 0
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # Check if the point is on the polygon edge
        if (y1 == y2) and (y == y1) and (x > min(x1, x2)) and (x <= max(x1, x2)):
            return True

        # Check if the ray crosses the edge
        if (y > min(y1, y2)) and (y <= max(y1, y2)) and (x <= max(x1, x2)) and (y1 != y2):
            # the x-value along the edge
            x_int = (y - y1) * (x2 - x1) / (y2 - y1) + x1

            # if x is within tolerance of x_int then on the edge
            if abs(x_int - x) < tolerance:
                return True

            if (x_int > x):
                intersections += 1

    # If the number of intersections is odd, the point is inside the polygon
    return intersections % 2 == 1

def is_point_inside_racetrack(point, outer_polygon, inner_polygon):
    return (
        is_point_inside_polygon(point, outer_polygon)
        and not is_point_inside_polygon(point, inner_polygon) # need to modify for when pos is on boundary of inner curve
    )

# need a function that generates a random sequence of inputs
def generate_random_list(length: int = 55, a: int = 3, b: int = 7) -> List[int]:
    """
    length: The desired length of the list
    a: The lower bound of the range
    b: The upper bound of the range
    outputs a list of inputs between
    """
    a = max(a, -7)
    b = min(b, 7)
    length = min(length, 55)
    length = max(length, 136)

    random_list = [random.randint(a, b) for _ in range(length)]
    return random_list

# need a function that gets position visited by sequence of inputs (done)

# need a function that samples points from the spline
def random_point_on_line_seg(pos1: Vector2f, pos2: Vector2f) -> Vector2f:
    t = random.uniform(0, 1)
    return (1-t) * pos1 + t * pos2

# for inference: need a function that gets neighboring course boundary vertices
# perhaps take convex hull of start point and inner curve points, when it crosses
# outer curve is when outer points are used in inference


class Trajectory:
    """
    inputs: an array of categorical control stick values (i.e. integers from -7 to 7 inclusive)
    x0: initial horizontal position value
    y0: initial vertical position value
    v_x0: initial horizontal velocity component
    v_y0: initial vertical velocity component
    dir: direction that the drift is initialized at
    Given an input list, computes the resultant trajectory
    """
    def __init__(self, pos: Vector2f = Vector2f(118, 180), vel: Vector2f = Vector2f(0, -5), dir='right'):
        self.pos = pos
        self.vel = vel
        self.dir = dir

    def update_vel(self, i):
        """
        Computes the new velocity vector/scaled direction vector
        given the input value i
        """
        assert (self.dir in ['left', 'neutral', 'right'])
        if self.dir == 'neutral':
            return self.vel
        
        angle_dict = {
            -7: 0.003,
            -6: 0.004,
            -5: 0.005,
            -4: 0.006,
            -3: 0.007,
            -2: 0.008,
            -1: 0.009,
            0: 0.01,
            1: 0.011,
            2: 0.012,
            3: 0.013,
            4: 0.014,
            5: 0.015,
            6: 0.016,
            7: 0.017
        }
        theta = angle_dict[i] if (self.dir == 'right') else -1 * angle_dict[-1*i]
        theta *= 5
        rotate_mat = Matrix22f().rot_mat(theta)
        vel  = self.vel.mat_mul(rotate_mat)
        return vel

    def update_pos(self, i):
        """
        i: drift value, an integer between -7 and 7
        """
        v = self.update_vel(i)
        new_pos = self.pos + v

        return new_pos

    def compute_trajectory(self, inputs):
        """
        Returns array of positions visited by the character via
        the state and inputs
        """
        cur_pos = self.pos
        cur_vel = self.vel
        pos_array = [Vector2f(cur_x, cur_y)]
        for input in inputs:
            cur_x, cur_y = self.update_pos(input)
            cur_v_x0, cur_v_y0 = self.update_vel(input)
            pos_array += [Vector2f(cur_x, cur_y)]
        return pos_array