import argparse
from genetic_optimization import Optimization
import datetime
from grid_search import GridSearch

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

    parameters = {
        'image': results.image,
        'population_size': 50,
        'polygons_count': 25,
        'time_to_run': [.1],
        'retain': [0.1, 0.2, 0.3, .5, .8],
        'mutate': [.5, .6, .9]
    }
    grid_search = GridSearch(**parameters)
    grid_search.search()

if __name__ == '__main__':
    main()