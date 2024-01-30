from dataclasses import dataclass
from typing import Union
import math

@dataclass
class Vector2f:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, rhs: "Vector2f"):
        return Vector2f(self.x + rhs.x, self.y + rhs.y)
    
    def __str__(self):
        return f"{[self.x, self.y]}"
    
    def __mul__(self, scalar: float) -> "Vector2f":
        scalar = float(scalar)
        return Vector2f(scalar * self.x, scalar * self.y)
    
    def __rmul__(self, scalar: float) -> "Vector2f":
        return self * scalar
    
    def dot(self, rhs: "Vector2f") -> float:
        return self.x * rhs.x + self.y * rhs.y
    
    def cross(self, rhs: "Vector2f") -> float:
        return self.x * rhs.y - rhs.x * self.y
    
    def mat_mul(self, left_mat: "Matrix22f") -> "Vector2f":
        first_row = Vector2f(left_mat.a, left_mat.b)
        last_row = Vector2f(left_mat.c, left_mat.d)
        return Vector2f(self.dot(first_row), self.dot(last_row))

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

pos = Vector2f(5, 5)
vel = Vector2f(2, 2)

my_mat = Matrix22f().rot_mat(0.52)
pos = Vector2f(1, 0)
print(f"Rotate: {pos.mat_mul(my_mat)}, scalar times vector: {5.0 * pos}")