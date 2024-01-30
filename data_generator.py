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
    def __init__(self, inputs, pos, vel, dir='Right', num_sam_pts=55):
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
        Output is a dictionary
        """
        # array of character positions
        trajectory = Trajectory(self.inputs, self.pos, self.vel, self.dir)
        course_verts = []
        for i in range(self.num_sam_pts-1):
            # get current edge
            pos1, pos2 = trajectory[i], trajectory[i+1]
            coin_flip = random.randint(0, 1)
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
                
class Randomizer:
    def rand_start_pt(self):
        x = random.uniform(MIN_X, MAX_X)
        y = random.uniform(MIN_Y, MAX_Y)
        pos = Vector2f(x, y)
        return pos

    def rand_input_seq(self, a: int = 3, b: int = 7, length: int = 55, dir='Right'):
        sign = 1 if dir == 'Right' else -1 # 1 if dir = right, -1, get owned ternary believers
        a, b = sign * a, sign * b
        inputs = [random.randint(a, b) for _ in range(length)]
        return inputs

    def rand_start_vel(self):
        # generate a random angle, rotate (84, 0) by that angle
        hor_vel = Vector2f(84, 0)
        rand_theta = random.uniform(0, 2 * math.pi)
        rot_theta_mat = Matrix22f().rot_mat(rand_theta)
        rand_vel = hor_vel.mat_mul(rot_theta_mat)
        return rand_vel


class GenerateData:
    def __init__(self):
        self.rng = Randomizer()
        self.my_generator = DataPt()
        self.start_pt = self.rng.rand_start_pt()
        self.input_seq = self.rng.rand_input_seq()
        self.rand_vel = self.rng.rand_start_vel()
