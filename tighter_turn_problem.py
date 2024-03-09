import math
from math_types import Vector2f, StandardLine, Circle, Matrix22f

class Strategy1:
    """
    We shall assume both points are below the circle, and
    pt1 is to the left
    """
    def __init__(self, circ: Circle, pt1: Vector2f, pt2: Vector2f):
        # need the two tangent points
        tan_pt1 = circ.tan_pt_from_pt(pt1) # left pt

        tan_pt2 = circ.tan_pt_from_pt(pt2, "right")

        # need the vectors from tan pts to center
        left_vec, right_vec = tan_pt1 - circ.cent, tan_pt2 - circ.cent

        # need angle between those vectors
        angle = left_vec.angle_between(right_vec)

        circle_arc = circ.arclength(angle)

        # need line segment lengths
        hyp1, hyp2 = pt1.eucl_dist(circ.cent), pt2.eucl_dist(circ.cent)
        seg_len1 = math.sqrt(hyp1 ** 2 - circ.r ** 2)
        seg_len2 = math.sqrt(hyp2 ** 2 - circ.r ** 2)
        pass