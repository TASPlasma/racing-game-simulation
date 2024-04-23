import random
import math
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
MIN_Y = -32768 # pretend this is Z
MAX_Y = 32767 #  pretend this is Z

# example x_vel, y_vel: 4.144531, -83.90503

class DataPt:
    """
    inputs: an input sequence
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
        trajectory = Trajectory(self.pos, self.vel, self.dir).compute_trajectory(self.inputs)
        course_verts = []
        for i in range(self.num_sam_pts-1):
            # get current edge
            pos1, pos2 = trajectory[i], trajectory[i+1]
            coin_flip = random.randint(0, 1) # roll to determine if edge is sampled from
            # sample from first and last edge guaranteed
            if (i == 0) or (i == self.num_sam_pts - 1) or coin_flip:
                # consider a sample from the bezier curve instead
                course_verts += [random_point_on_line_seg(pos1, pos2)]

        model_input_dict = {
            'pos': self.pos,
            'vel': self.vel,
            'dir': self.dir,
            'course_vertices': course_verts
        }
        model_label_dict = {
            'label': self.inputs
        }
        return {"inputs": model_input_dict, "label": model_label_dict}
    
    def __str__(self):
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


class GenerateData:
    def __init__(self):
        self.rng = Randomizer()
        self.start_pt = self.rng.rand_start_pt()
        self.input_seq = self.rng.rand_input_seq()
        self.rand_vel = self.rng.rand_start_vel()
        self.my_generator = DataPt(self.input_seq, self.start_pt, self.rand_vel)
        print(self.my_generator)

generator = GenerateData()
