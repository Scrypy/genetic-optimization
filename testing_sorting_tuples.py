from genetic_optimization import Optimization, Individual

img = Optimization('gothic.jpg')
img.generate_population(population_size=10, polygons_count=25)


graded = [(img.mse(individual), individual) for individual in img.population]
graded = sorted(graded, key=lambda x: x[0])
graded = [individual[0] for individual in graded]
assert graded == sorted(graded)
print("All is OK")
