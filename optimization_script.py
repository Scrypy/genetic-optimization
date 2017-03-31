import argparse


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

    img = Optimization(results.image, population_size=100, polygons_count=50)
    average_speed, improvement, fittest_individual_history = img.evolve_during(
        time_to_run=0.5,
        show_animation=True,
        save_images=False,
        retain=0.2,
        random_select=0.02,
        mutate=0.7
    )
    print('Improvement: ' + str(round(100 * improvement, 4)) + '%')
    print("\nTimestamp: ", datetime.datetime.now())
    print('Average speed ' + str(round(average_speed, 2)) + " gen/s")

if __name__ == '__main__':
    main()