import random
import math
import json
from math_types import Vector2f, Matrix22f
from simulate import Trajectory, random_point_on_line_seg
# can generate more data point in the following ways:
# 1. different input sequences
# 2. different sample points from cor. spline
# 3. different starting point
# 4. different start velocity
# 5. different number of sample pts <= number character pts

MIN_X = -32768
MAX_X = 32767
MIN_Y = -32768  # pretend this is Z
MAX_Y = 32767  # pretend this is Z

# example x_vel, y_vel: 4.144531, -83.90503


class DataPt:
    """
    inputs: an input sequence of control stick values
    pos: a starting position
    vel: a starting velocity
    dir: right, Left or Neutral (maybe?)
    num_sam_pts: number of points to sample from the spline
    """

    def __init__(self, inputs, pos, vel, dir='right', num_sam_pts=55):
        self.inputs = inputs
        self.pos = pos
        self.vel = vel
        self.dir = dir
        self.num_sam_pts = num_sam_pts

    def gen_data_pt(self):
        """
        A data point is:
        model input: relevant course vertices and state (pos, vel, dir)
        model label: the control inputs that take the turn 'optimally'
        Output is a dictionary {"inputs": state, "labels": ctrl_stick_vals}
        """
        # array of character positions
        trajectory = Trajectory(
            self.pos, self.vel, self.dir).compute_trajectory(self.inputs)
        course_verts = []
        for i in range(self.num_sam_pts-1):
            # get current edge
            pos1, pos2 = trajectory[i], trajectory[i+1]
            # roll to determine if edge is sampled from
            coin_flip = random.randint(0, 1)
            # sample from first and last edge guaranteed
            if (i == 0) or (i == self.num_sam_pts - 1) or coin_flip:
                # consider a sample from the bezier curve instead
                vert = random_point_on_line_seg(pos1, pos2)
                course_verts += [{'x': vert.x, 'y': vert.y}]

        model_input_dict = {
            'pos': {'x': self.pos.x, 'y': self.pos.y},
            'vel': {'x': self.vel.x, 'y': self.vel.y},
            'dir': self.dir,
            'course_vertices': course_verts
        }
        # can make all values positive here
        # so that model only outputs absolute value
        # and in inference just negate accordingly when the direction is left
        model_label_dict = {
            'label': self.inputs
        }
        return {"inputs": model_input_dict, "label": model_label_dict}

    def __str__(self):
        return f'{self.gen_data_pt()}'

    def __repr__(self):
        return f'{self.gen_data_pt()}'


class Randomizer:
    """
    Creates:
    -a random start point via rand_start_pt
    -a random input sequence via rand_input_seq
    -a random start velocity via rand_start_vel
    """

    def rand_start_pt(self):
        x = random.uniform(MIN_X, MAX_X)
        y = random.uniform(MIN_Y, MAX_Y)
        pos = Vector2f(x, y)
        return pos

    def rand_dir(self):
        options = ["left", "right"]
        # return random.choice(options)
        return "right"

    def rand_input_seq(self, a: int = 3, b: int = 7, length: int = 55, dir='right'):
        sign = 1 if dir == 'right' else -1
        a, b = sign * a, sign * b
        inputs = [random.randint(min(a, b), max(a, b)) for _ in range(length)]
        return inputs

    def rand_start_vel(self):
        hor_vel = Vector2f(84, 0)
        # generate a random angle, rotate (84, 0) by that angle
        rand_theta = random.uniform(0, 2 * math.pi)
        rot_theta_mat = Matrix22f().rot_mat(rand_theta)
        rand_vel = hor_vel.mat_mul(rot_theta_mat)
        return rand_vel


class DataGenerator:
    """
    Class that uses the DataPt and Randomizer class to generate a dataset
    has a method that outputs a list of any size,
    and a method that serves as a generator object
    """

    def __init__(self, num_data):
        self.rng = Randomizer()
        self.num_data = num_data

    def generate(self):
        data = []
        for i in range(self.num_data):
            if i % 1000 == 0:
                print(f"Progress: i: {i}, out of {self.num_data}")
            rand_dir = self.rng.rand_dir()
            start_pt = self.rng.rand_start_pt()
            start_vel = self.rng.rand_start_vel()
            input_seq = self.rng.rand_input_seq(dir=rand_dir)
            datum = DataPt(input_seq, start_pt, start_vel,
                           rand_dir).gen_data_pt()
            data += [datum]

        return data

    def __call__(self):
        """
        python generator version
        """
        for i in range(self.num_data):
            rand_dir = self.rng.rand_dir()
            start_pt = self.rng.rand_start_pt()
            start_vel = self.rng.rand_start_vel()
            input_seq = self.rng.rand_input_seq(dir=rand_dir)
            datum = DataPt(input_seq, start_pt, start_vel,
                           rand_dir).gen_data_pt()
            yield datum

    def save_data_to_file(self):
        data = self.generate()
        json_data = json.dumps(data, indent=4)
        with open('data1.json', 'w') as f:
            f.write(json_data)


generator = DataGenerator(10)


def num_pts_to_kbs(num_pts):
    return 5 * num_pts


def kb_to_num_pts(kb):
    return kb // 5


# print("num data points til 20 GB: ", kb_to_num_pts(20000000))

# data = generator.generate()
# print(data[0])
# generator.save_data_to_file()


# def test(m):
#     output = []
#     for i in range(m):
#         output += [i]

#     return output


# def test2(m):
#     for i in range(m):
#         yield i


# for datum in generator():
#     print(datum)

# for datum in data:
#     print(f"data point: {datum}")
