import numpy as np
import random
import time
import datetime
import copy
import os
import pickle
from PIL import Image, ImageDraw


class Individual(object):
    def __init__(self, shape, polygons_count, init_empty=False):
        self.shape = shape
        self.polygons_count = polygons_count
        self.score = None
        if not init_empty:
            self.polygons = self.generate_polygons()
        else:
            self.polygons = []

    def draw_individual(self):
        img = Image.new('RGB', (self.shape[1], self.shape[0]))
        draw = ImageDraw.Draw(img, 'RGBA')
        for polygon in self.polygons:
            draw.polygon(polygon.vertices, fill=polygon.color)
        return img

    def mse(self, image_array):
        if not self.score:
            individual_array = np.array(self.draw_individual(), dtype=np.int64)
            self.score = np.linalg.norm(np.subtract(individual_array, image_array))
        return self.score

    def generate_polygons(self):
        return [Polygon(self.shape) for _ in range(self.polygons_count)]

    def get_random_polygon(self):
        if not self.polygons:
            raise IndexError('Trying to get a random polygon without polygons.')
        return self.polygons[random.randint(0, self.polygons_count - 1)]

    def reset(self):
        self.score = None


class Polygon(object):
    def __init__(self, shape):
        # TODO: *Overkill* (Don't need to save the image shape in EVERY polygon)
        self.shape = shape
        self.vertices = self.generate_random_vertices(shape)
        self.color = self.random_color()

    def generate_random_vertices(self, shape):
        vertices = []
        vertices_count = random.randint(3, 5)
        for _ in range(vertices_count):
            a = random.randint(0, shape[1] - 1)
            b = random.randint(0, shape[0] - 1)
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

    def mutate_color(self, amplitude=3):
        channel = random.randint(0, 2)
        displacement = random.randint(-amplitude, amplitude)
        displacement = -2
        RGB = [self.color[0], self.color[1], self.color[2]]
        RGB[channel] = min(max(RGB[channel] + displacement, 0), 255)
        self.color = tuple(RGB + [50])


class Optimization(object):
    def __init__(self, path_to_image=None, population_size=None, polygons_count=None, init_empty=False):
        if init_empty:
            pass
        else:
            self.image_filename = os.path.basename(path_to_image)
            self.pickle_file = os.path.splitext(path_to_image)[0] + str('.pickle')
            self.image = Image.open(path_to_image).convert('RGB')
            self.image_as_array = np.array(self.image, dtype=np.int64)
            self.shape = self.image_as_array.shape
            self.generation = 0
            self.population = [Individual(self.shape, polygons_count) for _ in range(population_size)]
            self.fittest_individual = self.sort_population()[0]
            self.fittest_individual_history = []

    def save_image(self):
        img = self.fittest_individual.draw_individual()
        image_filename = 'image-{}.jpg'.format(self.generation)
        images_dir = 'images_' + os.path.splitext(self.image_filename)[0]
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        img.save(os.path.join(images_dir, image_filename))

    def resulting_image(self):
        img = self.fittest_individual.draw_individual()
        return img

    def grade(self):
        return sum([individual.mse(self.image_as_array) for individual in self.population]) / len(self.population)

    def sort_population(self):
        graded_list = [(individual.mse(self.image_as_array), individual) for individual in self.population]
        random.shuffle(graded_list)
        graded_list = sorted(graded_list, key=lambda x: x[0])
        graded_list = [individual[1] for individual in graded_list]
        self.fittest_individual = graded_list[0]
        return graded_list

    def sort_population_numpy(self):
        graded = np.array([[self.mse(individual), individual] for individual in self.population])
        random.shuffle(graded)
        graded = graded[graded[:, 0].argsort()]
        graded = [individual[1] for individual in graded]
        return graded

    def my_sort(self):
        graded = np.array([[self.mse(individual), individual] for individual in self.population])
        return 0

    def evolve(self, retain, random_select, mutate):

        # Selection of the fittest
        # TODO: Test which sort is faster (np or built in) and check if they sort by the same array
        # TODO: write its own function
        # sorted_list = self.sort_population_numpy()
        sorted_list = self.sort_population()
        # sorted_numpy = self.sort_population_numpy()
        retain_lenght = int(len(sorted_list) * retain)
        new_population = sorted_list[:retain_lenght]

        # Random selection
        for individual in sorted_list[retain_lenght:]:
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
            if male != female:
                male = new_population[male]
                female = new_population[female]
                # half = int(len(male.polygons) / 2)
                polygons_count = len(female.polygons)
                child = Individual(self.shape, polygons_count, init_empty=True)
                # child.polygons = male.polygons[:half] + female.polygons[half:]
                child.polygons = copy.deepcopy(female.polygons)
                children.append(copy.deepcopy(child))
                # children.append(copy.deepcopy(best_individual))

        # Mutation
        for child in children:
            if mutate > random.random():
                child.get_random_polygon().mutate_color()
                child.reset()
            if mutate > random.random():
                child.get_random_polygon().mutate_vertex()
                child.reset()

        new_population.extend(children)
        self.population = new_population
        self.generation += 1
        return 0

    def save_status(self):
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self, f)

    def load_status(self, filename):
        print('Resuming...')
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def picture_time(self, start, time_to_run, photograms_count, photograms):
        t = time_to_run * photograms_count / float(photograms)
        return datetime.datetime.now() > start + datetime.timedelta(0, 60 * t)

    def evolve_during(self, time_to_run=0.5, frames=10, show_animation=False, save_images=False,
                      retain=0.2, random_select=0.02, mutate=0.7, verbose=False,
                      save_progress=False):
        # Setting times
        starting_time = datetime.datetime.now()
        ending_time = starting_time + datetime.timedelta(0, 60 * time_to_run)

        # Lists of data
        images_list = []
        generation_count = 0
        photograms_count = 1

        self.save_image()
        m_list = []
        print('Running for ' + str(time_to_run) + ' min')
        print("Completion time: ", ending_time)
        print('-----')
        start = time.time()

        while datetime.datetime.now() < ending_time:
            generation_count += 1
            self.evolve(retain, random_select, mutate)
            if self.picture_time(starting_time, time_to_run, photograms_count, frames):
                photograms_count += 1

                if show_animation:
                    images_list.append(self.resulting_image())
                if save_images:
                    self.save_image()

                fittest = round(self.fittest_individual.mse(self.image_as_array), 2)
                self.fittest_individual_history.append(fittest)
                if verbose:
                    print(fittest)

        if save_progress:
            self.save_status()

        if show_animation:
            if images_list:
                images_list[0].save('animation.gif', save_all=True, append_images=images_list)

        self.save_image()
        average_speed = generation_count / (time.time() - start)
        improvement = (self.fittest_individual_history[0]
                    - self.fittest_individual_history[-1]) / self.fittest_individual_history[0]
        return average_speed, improvement, self.fittest_individual_history, self.generation

