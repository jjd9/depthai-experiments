import numpy as np
from feature import FeatureSet

class Bucket:
    def __init__(self, size):
        self.max_size = size
        self.features = FeatureSet()

    def size(self):
        return self.features.size()

    def add_feature(self, point, age):
        # won't add feature with age > 10
        age_threshold = 10
        if age < age_threshold:
            # insert any feature before bucket is full
            if self.size() < self.max_size:
                self.features.points.append(point)
                self.features.ages.append(age)
            else:
                # insert feature with old age and remove youngest one
                age_min_idx = np.argmin(self.features.ages)
                self.features.points[age_min_idx] = point
                self.features.ages[age_min_idx] = age

    def get_features(self, current_features):
        current_features.points += self.features.points
        current_features.ages += self.features.ages
