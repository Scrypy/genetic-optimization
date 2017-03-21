from PIL import Image, ImageDraw, ImageChops
# from skimage.measure import structural_similarity as ssim
import numpy as np
import random
import argparse
import time


class GeneticOptimization(object):

    def __init__(self, path_to_image):
        self.image = Image.open(path_to_image).convert('RGB')
        self.shape = np.array(self.image).shape
        self.resulting_image = Image.fromarray(np.zeros(self.shape), 'RGB')

    def save_image(self, generation_count):
        self.draw_population()
        image_fname = 'image-{}.jpg'.format(generation_count)
        self.resulting_image.save(image_fname)

    def generate_population(self, number_of_individuals, vertices_count=None):
        self.population = [Individual(self.shape, vertices_count) for _ in range(number_of_individuals)]

    def draw_population(self):
        new_image = Image.fromarray(np.zeros(self.shape), 'RGB')
        draw = ImageDraw.Draw(new_image, 'RGBA')
        for individual in self.population:
            draw.polygon(individual.vertices, fill=individual.color)
        self.resulting_image = new_image

    def draw_individual(self, individual):
        img = Image.fromarray(np.zeros(self.shape), 'RGB')
        draw = ImageDraw.Draw(img, 'RGBA')
        draw.polygon(individual.vertices, fill=individual.color)
        return img

    def fitness(self, individual):
        diff = ImageChops.difference(self.draw_individual(individual), self.image)
        diff = np.array(diff)
        diff = np.linalg.norm(diff.ravel())
        return diff

    def mse(self, individual):
        diff = ImageChops.difference(
            self.draw_individual(individual),
            self.image
            )
        diff = np.array(diff).ravel()
        diff = sum(diff ** 2) / len(diff)
        return diff

    def grade(self):
        return sum([self.fitness(individual) for individual in self.population]) / len(self.population)

    def evolve(self, retain=0.2, random_select=0.02, mutate=0.1):

        # Selection of the fittest
        graded = [(self.mse(individual), individual) for individual in self.population]
        graded = sorted(graded, key=lambda x: x[0])
        graded = [individual[1] for individual in graded]
        retain_lenght = int(len(graded) * retain)
        new_population = graded[:retain_lenght]

        # Random selection
        for individual in graded[retain_lenght:]:
            if random.random() < random_select:
                new_population.append(individual)

        # Mutation
        for individual in new_population:
            if mutate > random.random():
                individual.mutate_vertex()
            if mutate > random.random():
                individual.mutate_color()

        # Breeding
        if len(self.population) < 20:
            print("Population is too small for breeding")
        necessary_children = len(self.population) - len(new_population)
        children = []
        while len(children) < necessary_children:
            male = random.randint(0, len(new_population) - 1)
            female = random.randint(0, len(new_population) - 1)
            if male != female:
                male = new_population[male]
                female = new_population[female]
                half = int(len(male.vertices) / 2)
                child = Individual(self.shape, init_empty=True)
                child.vertices = male.vertices[:half] + female.vertices[half:]
                child.color = male.color
                children.append(child)
        new_population.extend(children)
        self.population = new_population
        return 0

class Individual(object):

    def __init__(self, polygons_count=10, genes=3, start_empty=False):
        self.genes = genes
        if start_empty:
            self.polygons = []
        else:
            self.polygons = self.generate_polygons(polygons_count, genes)

    def generate_polygons(polygons_count, genes):
        polygons_list = []
        for polygon in polygons_count:
            polygons_list.append(Polygon())

class Polygon(object):

    def __init__(self, shape, max, min, vertices_count=None, init_empty=False):
        self.shape = shape
        self.vertices = []
        self.color = None
        if not init_empty:
            self.vertices = self.generate_random_vertices(vertices_count)
            self.color = self.random_color()

    def random_color(self):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return r, g, b, 50

    def generate_random_vertices(self, vertices_count=None):
        if not vertices_count:
            vertices_count = random.randint(3, 5)
        vertices = []
        for _ in range(vertices_count):
            a = random.randint(0, self.shape[1] - 1)
            b = random.randint(0, self.shape[0] - 1)
            vertices.append((a, b))
        return vertices

    def mutate_vertex(self):
        gen = random.randint(0, len(self.vertices) - 1)
        a = random.randint(0, self.shape[1] - 1)
        b = random.randint(0, self.shape[0] - 1)
        self.vertices[gen] = a, b

    def mutate_color(self):
        self.color = self.random_color()


def show_image(nparray, shape, mode='RGBA'):
    image = Image.fromarray(nparray.reshape(shape), mode)
    image.show()


def save_as_gif(images_list):
    images_list[0].save('temp.gif', save_all=True, append_images=images_list)


def main():
    parser = argparse.ArgumentParser(
        description='Optimizing image representation?',
        prog='ImageCompression'
        )

    parser.add_argument(
        'image',
        type=str,
        help='Path to image e.g. /images/pic.jpg'
        )
    results = parser.parse_args()

    population_size = 50
    show_animation = False
    save_image_every = 5
    images_list = []

    img = GeneticOptimization(results.image)
    img.generate_population(population_size)
    img.draw_population()
    images_list.append(img.resulting_image)

    start = time.time()
    for generation_count in range(2000):
        img.evolve(retain=0.2, random_select=0.06, mutate=0.2)
        if show_animation:
            img.draw_population()
            images_list.append(img.resulting_image)
        if generation_count % save_image_every == 0:
            img.save_image(generation_count)
            print(str(round((time.time() - start) / 60, 2)) + " min")
    if show_animation:
        images_list[0].save('temp.gif', save_all=True, append_images=images_list)

if __name__ == "__main__":
    main()
