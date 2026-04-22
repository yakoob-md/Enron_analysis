# multiclass_pipeline/preprocessing/label_encoder.py
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os

class MultiClassLabelEncoder:
    def __init__(self, class_names):
        self.le = LabelEncoder()
        self.le.fit(class_names)
        self.class_names = class_names

    def encode(self, labels):
        return self.le.transform(labels)

    def decode(self, indices):
        return self.le.inverse_transform(indices)

    def get_weights(self, labels):
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(labels)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        return dict(zip(classes, weights))

    def save(self, path):
        joblib.dump(self.le, path)

    @classmethod
    def load(cls, path):
        le = joblib.load(path)
        instance = cls(le.classes_)
        instance.le = le
        return instance
