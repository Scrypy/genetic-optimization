import numpy as np
import random
import argparse
import time
import datetime
import line_profiler
from PIL import Image, ImageDraw


class Optimization(object):
    def __init__(self, path_to_image):
        self.image = Image.open(path_to_image).convert('RGB')
        self.image_asarray = np.array(self.image, dtype=np.int64)
        self.shape = self.image_asarray.shape
        self.population = []
        self.generation = 0

    def fittest_individual(self):
        graded = [(self.mse(individual), individual) for individual in self.population]
        graded = sorted(graded, key=lambda x: x[0])
        graded = [individual[1] for individual in graded]
        return graded[0]

    def save_image(self):
        img = self.draw_individual(self.fittest_individual())
        image_filename = 'image-{}.jpg'.format(self.generation)
        img.save(image_filename)

    def resulting_image(self):
        img = self.draw_individual(self.fittest_individual())
        image_filename = 'image-{}.jpg'.format(self.generation)
        return img

    def generate_population(self, population_size, polygons_count):
        pop = []
        for _ in range(population_size):
            pop.append(Individual(self.shape, polygons_count))
        self.population = pop

    def draw_individual(self, individual):
        img = Image.fromarray(np.zeros(self.shape), 'RGB')
        draw = ImageDraw.Draw(img, 'RGBA')
        for polygon in individual.polygons:
            draw.polygon(polygon.vertices, fill=polygon.color)
        return img

    def mse(self, individual):
        if individual.score:
            return individual.score
        individual_array = np.array(self.draw_individual(individual), dtype=np.int64)
        image_array = self.image_asarray
        # diff = np.absolute(np.subtract(individual_array, image_array)).sum()
        # diff = np.linalg.norm(np.subtract(individual_array, image_array))
        diff = np.power(np.subtract(individual_array, image_array), 2).sum()
        diff = np.divide(diff, float(self.shape[0] * self.shape[1]))
        individual.score = diff
        return diff

    def grade(self):
        return sum([self.mse(individual) for individual in self.population]) / len(self.population)

    def evolve(self, retain=0.2, random_select=0.01, mutate=0.005):

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
                p = random.randint(0, len(individual.polygons) - 1)
                p = individual.polygons[p]
                p.mutate_vertex()
                individual.score = None
            if mutate > random.random():
                p = random.randint(0, len(individual.polygons) - 1)
                p = individual.polygons[p]
                p.mutate_color()
                individual.score = None

        # Breeding
        if len(self.population) < 10:
            raise ValueError('Population is too small for breeding')
        necessary_children = len(self.population) - len(new_population)
        children = []
        while len(children) < necessary_children:
            male = random.randint(0, len(new_population) - 1)
            female = random.randint(0, len(new_population) - 1)
            if male != female:
                male = new_population[male]
                female = new_population[female]
                half = int(len(male.polygons) / 2)
                polygons_count = len(male.polygons)
                child = Individual(self.shape, polygons_count, init_empty=True)
                child.polygons = male.polygons[:half] + female.polygons[half:]
                children.append(child)
        new_population.extend(children)
        self.population = new_population
        self.generation += 1
        return 0

    def evolve_during(self, time_to_run=0.5, save_image_every=1500, show_animation=False):
        self.save_image()
        t = datetime.timedelta(0, 60 * time_to_run)
        starting_time = datetime.datetime.now()
        ending_time = starting_time + t
        images_list = []
        population_history = []
        print('Running for ' + str(time_to_run) + ' min')
        print("Completion time: ", ending_time)
        print('-----')
        start = time.time()
        generation_count = 0
        while(datetime.datetime.now() < ending_time):
            generation_count += 1
            self.evolve()
            if self.generation % save_image_every == 0:
                if show_animation:
                    images_list.append(self.resulting_image())
                self.save_image()
                speed = generation_count / (time.time() - start)
                population_history.append(self.grade())
        if show_animation:
            if images_list:
                images_list[0].save('animation.gif', save_all=True, append_images=images_list)
        self.save_image()
        average_speed = self.generation / (time.time() - start)
        print('Average speed' + str(round(average_speed, 2)) + " gen/s")
        print("\nTimestamp: ", datetime.datetime.now())
        print(population_history)


class Individual(object):
    def __init__(self, shape, polygons_count, init_empty=False):
        self.shape = shape
        self.polygons_count = polygons_count
        self.score = None
        if not init_empty:
            self.polygons = self.generate_polygons(shape, polygons_count)
        else:
            self.polygons = []

    def generate_polygons(self, shape, polygons_count):
        return [Polygon(shape) for _ in range(polygons_count)]


class Polygon(object):
    def __init__(self, shape):
        self.shape = shape
        self.vertices = self.generate_random_vertices(shape)
        self.color = self.random_color()

    def generate_random_vertices(self, shape):
        vertices = []
        vertices_count = random.randint(3, 5)
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

    def random_color(self):
        return tuple([random.randint(0, 255) for _ in range(3)] + [50])

    def mutate_color(self):
        self.color = self.random_color()


def main():
    parser = argparse.ArgumentParser(
        description='Optimizing number of polygons',
        prog='genetic-optimization'
    )
    parser.add_argument(
        'image',
        type=str,
        help='Path to image e.g. /images/pic.jpg'
    )
    results = parser.parse_args()

    img = Optimization(results.image)
    img.generate_population(population_size=10, polygons_count=25)
    img.evolve_during(time_to_run=5, save_image_every=1500, show_animation=False)


if __name__ == '__main__':
    main()
