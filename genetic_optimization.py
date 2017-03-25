import numpy as np
import random
import argparse
import time
import datetime
import copy
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
        # diff = np.linalg.norm(np.subtract(individual_array, image_array).reshape((3, -1)))
        # diff = np.linalg.norm(np.subtract(individual_array, image_array))
        # diff = np.power(np.subtract(individual_array, image_array), 2).sum()
        diff = np.absolute(np.subtract(individual_array, image_array)).sum()
        individual.score = diff
        return diff

    def grade(self):
        return sum([self.mse(individual) for individual in self.population]) / len(self.population)

    def evolve(self, retain=0.2, random_select=0.00, mutate=1.0):

        # Selection of the fittest
        graded = [(self.mse(individual), individual) for individual in self.population]
        graded = sorted(graded, key=lambda x: x[0])
        graded = [individual[1] for individual in graded]
        retain_lenght = int(len(graded) * retain)
        new_population = copy.deepcopy(graded[:retain_lenght])

        # Random selection
        for individual in graded[retain_lenght:]:
            if random.random() < random_select:
                new_population.append(copy.deepcopy(individual))

        # Breeding
        if len(self.population) < 10:
            raise ValueError('Population is too small for breeding')
        necessary_children = len(self.population) - len(new_population)
        children = []
        while len(children) < necessary_children:
            male = random.randint(0, len(new_population) - 1)
            female = random.randint(0, len(new_population) - 1)
            # if male != female:
            if True:
                male = new_population[male]
                female = new_population[female]
                half = int(len(male.polygons) / 2)
                polygons_count = len(female.polygons)
                child = Individual(self.shape, polygons_count, init_empty=True)
                # child.polygons = male.polygons[:half] + female.polygons[half:]
                child.polygons = copy.deepcopy(female.polygons)
                # children.append(copy.deepcopy(child))
                children.append(copy.deepcopy(new_population[0]))

        # Mutation
        for individual in children:
            if mutate > random.random():
                p = random.randint(0, len(individual.polygons) - 1)
                p = individual.polygons[p]
                # p.mutate_vertex()
                individual.score = None
            if mutate > random.random():
                p = random.randint(0, len(individual.polygons) - 1)
                p = individual.polygons[p]
                p.mutate_color()
                individual.score = None

        new_population.extend(children)
        self.population = new_population
        self.generation += 1
        return 0

    def picture_time(self, start, time_to_run, photograms_count, photograms):
        return datetime.datetime.now() > start + datetime.timedelta(0, 60 * time_to_run * photograms_count / float(photograms))

    def evolve_during(self, time_to_run=0.5, save_image_every=1500, photograms=10, show_animation=False):
        # Setting times
        starting_time = datetime.datetime.now()
        ending_time = starting_time + datetime.timedelta(0, 60 * time_to_run)
        picture_time = datetime.timedelta(time_to_run / float(photograms))
        picture_time = starting_time + datetime.timedelta(0, 60 * time_to_run / float(photograms))

        # Lists of data
        images_list = []
        population_history = []
        fittest_individual_history = []
        generation_count = 0
        photograms_count = 1

        self.save_image()
        print('Running for ' + str(time_to_run) + ' min')
        print("Completion time: ", ending_time)
        print('-----')
        start = time.time()
        while(datetime.datetime.now() < ending_time):
            generation_count += 1
            self.evolve()
            if self.picture_time(starting_time, time_to_run, photograms_count, photograms):
                photograms_count += 1
                if show_animation:
                    images_list.append(self.resulting_image())
                self.save_image()
                speed = generation_count / (time.time() - start)
                population_history.append(self.grade())
                fittest_individual = round(self.mse(self.fittest_individual()), 2)
                fittest_individual_history.append(fittest_individual)
        if show_animation:
            if images_list:
                images_list[0].save('animation.gif', save_all=True, append_images=images_list)
        self.save_image()
        average_speed = self.generation / (time.time() - start)
        print('Average speed ' + str(round(average_speed, 2)) + " gen/s")
        print("\nTimestamp: ", datetime.datetime.now())
        # print(population_history)
        print(fittest_individual_history)
        print('Improvement: ', (fittest_individual_history[0]-fittest_individual_history[-1])/fittest_individual_history[0])


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

    def mutate_vertex(self, amplitude=5):
        gen = random.randint(0, len(self.vertices) - 1)
        direction = random.randint(0, 1)
        displacement = random.randint(-amplitude, amplitude)
        new_vertex = [self.vertices[gen][0], self.vertices[gen][1]]
        new_vertex[direction] = min(max(new_vertex[direction] + displacement, 0), self.shape[1 - direction] - 1)
        self.vertices[gen] = tuple(new_vertex)

    def random_color(self):
        return tuple([random.randint(0, 255) for _ in range(3)] + [50])

    def mutate_color(self, amplitude=2):
        channel = random.randint(0, 2)
        displacement = random.randint(-amplitude, 0)
        # displacement = random.randint(-amplitude, amplitude)
        RGB = [self.color[0], self.color[1], self.color[2]]
        RGB[channel] = min(max(RGB[channel] + displacement, 0), 255)
        self.color = tuple(RGB + [50])


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
    img.generate_population(population_size=100, polygons_count=25)
    img.evolve_during(time_to_run=2, save_image_every=10, show_animation=True)

if __name__ == '__main__':
    main()
