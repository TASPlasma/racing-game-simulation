from dataclasses import dataclass
from typing import Union
import math

@dataclass
class Vector2f:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, rhs: "Vector2f") -> "Vector2f":
        return Vector2f(self.x + rhs.x, self.y + rhs.y)
    
    def __sub__(self, rhs: "Vector2f") -> "Vector2f":
        return Vector2f(self.x - rhs.x, self.y - rhs.y)

    def __str__(self):
        return f"{[self.x, self.y]}"
    
    def __mul__(self, scalar: float) -> "Vector2f":
        scalar = float(scalar)
        return Vector2f(scalar * self.x, scalar * self.y)
    
    def __rmul__(self, scalar: float) -> "Vector2f":
        return self * scalar
    
    def dot(self, rhs: "Vector2f") -> float:
        return self.x * rhs.x + self.y * rhs.y
    
    def norm(self) -> float:
        return math.sqrt(self.dot(self))
    
    def cross(self, rhs: "Vector2f") -> float:
        return self.x * rhs.y - rhs.x * self.y
    
    def mat_mul(self, left_mat: "Matrix22f") -> "Vector2f":
        first_row = Vector2f(left_mat.a, left_mat.b)
        last_row = Vector2f(left_mat.c, left_mat.d)
        return Vector2f(self.dot(first_row), self.dot(last_row))
    
    def eucl_dist(self, rhs: "Vector2f") -> float:
        """
        Returns the euclidean distance from the points
        self and rhs
        """
        diff = rhs - self
        return diff.norm()
    
    def angle_between(self, rhs: "Vector2f") -> float:
        # v dot u = ||v|| ||u|| cos(theta)
        # theta = arccos(v dot u / (||v|| ||u||))
        num = self.dot(rhs)
        denom = self.norm() * rhs.norm()
        return math.acos(num / denom)


@dataclass
class Matrix22f:
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0

    def __add__(self, rhs: "Matrix22f"):
        return Matrix22f(self.a + rhs.a, self.b + rhs.b, self.c + rhs.c, self.d + rhs.d)
    
    def __str__(self):
        return f"[{self.a} {self.b}]\n[{self.c} {self.d}]"
    
    def rot_mat(self, theta: float = 0.0) -> None:
        self.a = math.cos(theta)
        self.b = -math.sin(theta)
        self.c = math.sin(theta)
        self.d = math.cos(theta)
        return self
    
    def first_col(self) -> Vector2f:
        return Vector2f(self.a, self.c)
    
    def second_col(self) -> Vector2f:
        return Vector2f(self.b, self.d)
    
@dataclass
class Circle:
    r: float = 1.0
    cent: Vector2f = Vector2f(0, 0)

    def pt_at_angle(self, theta: float = 0.0) -> Vector2f:
        return Vector2f(self.r * math.cos(theta), self.r * math.sin(theta))
    
    def angle_at_pt(self, pt: Vector2f) -> float:
        """
        Returns the angle theta for which
        pt lies on the circle
        """
        delta = pt - self.cent
        theta = math.atan2(delta.y, delta.x)

        return theta + 2 * math.pi if theta < 0 else theta
    
    def __str__(self):
        return f"(x-{self.cent.x})^2 + (x-{self.cent.y})^2 = {self.r}^2"
    
    def tan_pt_from_pt(self, pt: Vector2f, side: str = "left") -> Vector2f:
        """Given a point outside the circle, 
        computes the point P on the circle
        such that the secant line from pt to P
        and the tangent line at P are the same
        """
        
        a = pt.x
        b = pt.y
        s = self.cent.x
        k = self.cent.y

        scalar = 1 if side == "left" else -1

        # eq of tangent line in pt slope is y-b = num/denom(x-a)
        num = (a-s)*(b-k) + scalar * math.sqrt(((a-s)*(b-k)) ** 2 - ((a-s) ** 2 - self.r ** 2)*((b-k) ** 2 - self.r ** 2))
        denom = ((a-s) ** 2 - self.r **2)

        # standard form of tangent line
        line = StandardLine().standard_line_from_pt_slope(pt, num, denom)

        # need distance from (a, b) to tangent pt
        hyp = pt.eucl_dist(self.cent)
        dist = math.sqrt(hyp ** 2 - self.r ** 2)

        pts = line.pts_dist_from_pt(pt, dist)

        cand1, cand2 = pts.first_col(), pts.second_col()

        if cand1.eucl_dist(self.cent) < cand2.eucl_dist(self.cent):
            return cand1

        return cand2
    
    def arclength(self, angle: float) -> float:
        """
        Computes arclength from given angle
        """
        return self.r * angle

@dataclass
class StandardLine:
    a: float = 0.0
    b: float = 1.0
    c: float = 0.0

    def __str__(self):
        return f"{self.a}x + {self.b}y = c"
    
    def slope(self):
        return float('inf') if self.b == 0 else -self.a / self.b
    
    def standard_line_from_pt_slope(self, pt: Vector2f, num: float, denom: float) -> "StandardLine":
        """
        Given a line of the form
        y-pt.y = num/denom(x-pt.x), converts
        to a line of the form
        Ax + By = C
        returns a line object
        """
        return StandardLine(-num, denom, -num*pt.x + denom*pt.y)
    
    def pts_dist_from_pt(self, pt: Vector2f, dist: float = 0) -> Matrix22f:
        """
        Returns a matrix whose columns are the two points
        on the line that are dist-distance away from the input pt
        """
        if self.slope() == float('inf'):
            return Matrix22f(pt.x, pt.x, pt.y + dist, pt.y - dist)
        
        delta_x = math.sqrt(dist ** 2 / (1 + self.slope ** 2))
        
        delta = Vector2f(delta_x, self.slope * delta_x)

        return Matrix22f(pt.x + delta.x, pt.x-delta.x, pt.y + delta.y, pt.y-delta.y)

pos = Vector2f(5, 5)
vel = Vector2f(2, 2)

my_mat = Matrix22f().rot_mat(0.52)
pos = Vector2f(1, 0)
print(f"Rotate: {pos.mat_mul(my_mat)}, scalar times vector: {5.0 * pos}")