from sklearn.utils.extmath import cartesian
import numpy as np
from genetic_optimization import Optimization

class GridSearch(object):
    def __init__(self, **data):
        self.image_path = data.pop('image')
        self.population_size = data.pop('population_size')
        self.polygons_count = data.pop('polygons_count')
        self.data = data

    def search(self):
        arg_keywords = np.array([key for key, _ in self.data.items()])
        combinations = cartesian([self.data[x] for x in arg_keywords])
        print(combinations)
        print(combinations.shape[0], 'combinations.')
        results = []
        dictionaries_list = []

        for parameter_combination in combinations:
            # Initializing Population
            optimization = Optimization(
                self.image_path,
                population_size=self.population_size,
                polygons_count=self.polygons_count
            )

            dict = {}
            for column, value in enumerate(parameter_combination):
                dict[arg_keywords[column]] = value
            dictionaries_list.append(dict)
            results.append(optimization.evolve_during(**dict)[1])

        # Ordering results
        order = sorted(range(len(results)), key=lambda k: results[k], reverse=True)
        dictionaries_list = [dictionaries_list[i] for i in order]
        results = [results[i] for i in order]

        for index, result in enumerate(results):
            print('Improvement ' + str(result), '\tParameters: ' + str(dictionaries_list[index]))