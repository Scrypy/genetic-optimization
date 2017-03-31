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

    # optimization = Optimization(results.image, population_size=100, polygons_count=50)
    optimization = Optimization(init_empty=True)
    optimization = optimization.load_status('images/gothic.pickle')
    average_speed, improvement, fittest_individual_history, generation = optimization.evolve_during(
        time_to_run=10,
        show_animation=True,
        save_images=True,
        retain=0.2,
        random_select=0.02,
        mutate=0.9,
        save_progress=True
    )
    print('Improvement: ' + str(round(100 * improvement, 4)) + '%')
    print("\nTimestamp: ", datetime.datetime.now())
    print('Average speed ' + str(round(average_speed, 2)) + " gen/s")
    print('Generation: ' + str(generation))

if __name__ == '__main__':
    main()