import numpy as np
from PIL import Image
from .. import genetic_optimization


def mse(self, individual):
    individual_array = np.array(self.draw_individual(individual), dtype=np.int64)
    image_array = self.image_asarray
    diff = np.absolute(np.subtract(individual_array, image_array)).sum()
    individual.score = diff
    return diff


img = np.array(Image.open('test_small2x2.jpg').convert('RGB'))
img2 = np.array(Image.open('test_small2x2-2.jpg').convert('RGB'))
print(img)
