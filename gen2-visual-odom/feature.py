class FeaturePoint:
    def __init__(self):
        self.point = None
        self.id = None
        self.age = None

class FeatureSet:
    def __init__(self):
        self.points = []
        self.ages = []

    def size(self):
        return len(self.points)

    def clear(self):
        self.points = []
        self.ages = []
