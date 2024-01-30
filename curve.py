class Curve:
    def __init__(self, color, points):
        self.color = color
        self.points = points

    def __iter__(self):
        return iter(self.points)

    def clear(self):
        self.points = []