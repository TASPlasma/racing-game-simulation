import math
from math_types import Vector2f, StandardLine, Circle, Matrix22f



class ProblemObject:
    """
    We shall assume both points are below the circle, and
    pt1 is to the left
    """
    def __init__(self, circ: Circle, pt1: Vector2f, pt2: Vector2f):
        self.circ = circ
        self.pt1 = pt1
        self.pt2 = pt2
        print(f"This turn takes {self.length_of_opt_turn()} units.")

    def length_of_opt_turn(self):
        # need the two tangent points
        tan_pt1 = self.circ.tan_pt_from_pt(self.pt1) # left pt

        tan_pt2 = self.circ.tan_pt_from_pt(self.pt2, "right")

        # need the vectors from tan pts to center
        left_vec, right_vec = tan_pt1 - self.circ.cent, tan_pt2 - self.circ.cent

        # need angle between those vectors
        angle = left_vec.angle_between(right_vec)

        circle_arc = self.circ.arclength(angle)

        # need line segment lengths
        hyp1 = self.pt1.eucl_dist(self.circ.cent)
        hyp2 = self.pt2.eucl_dist(self.circ.cent)
        seg_len1 = math.sqrt(hyp1 ** 2 - self.circ.r ** 2)
        seg_len2 = math.sqrt(hyp2 ** 2 - self.circ.r ** 2)

        return seg_len1 + circle_arc + seg_len2
    
class Strategy1:
    """
    Places the center of the larger circle according to strategy 1

    """
    def __init__(self, pt1: Vector2f, pt2: Vector2f, circ: Circle, R):
        """
        Need the line seg from pt1 to pt2,
        the bisector from smaller circ to that line seg

        Need location on bisector of center of larger circle
        """
        self.pt1 = pt1
        self.pt2 = pt2
        self.circ = circ
        self.R = R

        
    def large_circle(self) -> Circle:
        mid_pt = (self.pt1 + self.pt2) * 0.5
        dist = mid_pt.eucl_dist(self.circ.cent) + self.circ.r

        # define line from circ.cent to mid_pt
        num = mid_pt.y - self.circ.y
        denom = mid_pt.x - self.circ.x

        line = StandardLine().standard_line_from_pt_slope(
            num, 
            denom, 
            self.circ.cent
            )

        pts = line.pts_dist_from_pt(self.pt1, dist)

        cand1, cand2 = pts.first_col(), pts.second_col()
        cand = cand2

        if cand1.eucl_dist(self.circ.cent) < cand2.eucl_dist(self.circ.cent):
            cand = cand1

        # need angle that this pt occurs at:
        angle = self.circ.angle_at_pt(cand)
        
        # need center of outer circle:
        x_coord = self.circ.cent.x + (self.circ.r - self.R) * math.cos(angle)
        y_coord = self.circ.cent.y + (self.circ.r - self.R) * math.cos(angle)

        outer_circ = Circle(self.R, Vector2f(x_coord, y_coord))

        return outer_circ
        
        

class Strategy2:
    """
    Places the center of the larger circle according to strategy 2

    """
    def __init__(self):
        """
        Need tangential lines from pt1 and pt2 to smaller circle
        Create line segment joining these points
        Bisect that line segment via the center of the smaller circle
        """