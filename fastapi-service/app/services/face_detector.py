import cv2
import numpy as np
from typing import List, Tuple

class HaarFaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size: Tuple[int, int] = (50, 50)):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect(self, gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Mendeteksi posisi koordinat wajah pada image grayscale."""
        faces = self.cascade.detectMultiScale(
            gray_img,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        return faces