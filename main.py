# import numpy as np
# import itertools

# def get_input_coords(planet):
#     x = float(input(f"Enter the x-coordinate for {planet}: "))
#     y = float(input(f"Enter the y-coordinate for {planet}: "))
#     return np.array([x, y])

# def calc_lagrange_points(planet_coords):
#     d = np.sqrt(planet_coords[0]**2 + planet_coords[1]**2)
#     angle = np.radians(60)
#     l4 = planet_coords + d * np.array([np.cos(angle), np.sin(angle)])
#     l5 = planet_coords + d * np.array([np.cos(-angle), np.sin(-angle)])
#     return l4, l5

# def cartesian_to_heliocentric(coords):
#     mu = np.sqrt(coords[0]**2 + coords[1]**2)
#     nu = np.arctan2(coords[1], coords[0])
#     return mu, nu

# def calc_distance(coords1, coords2):
#     mu1, nu1 = cartesian_to_heliocentric(coords1)
#     mu2, nu2 = cartesian_to_heliocentric(coords2)
#     return np.sqrt(mu1**2 + mu2**2 - 2*mu1*mu2*np.cos(nu1-nu2))

# def find_shortest_path(planets):
#     paths = [
#         ('Earth', 'Earth L4', 'Mars L4', 'Jupiter'),
#         ('Earth', 'Earth L4', 'Mars L5', 'Jupiter'),
#         ('Earth', 'Earth L4', 'Venus L4', 'Jupiter'),
#         ('Earth', 'Earth L4', 'Venus L5', 'Jupiter'),
#         ('Earth', 'Earth L4', 'Jupiter'),
#         ('Earth', 'Earth L5', 'Mars L4', 'Jupiter'),
#         ('Earth', 'Earth L5', 'Mars L5', 'Jupiter'),
#         ('Earth', 'Earth L5', 'Venus L4', 'Jupiter'),
#         ('Earth', 'Earth L5', 'Venus L5', 'Jupiter'),
#         ('Earth', 'Earth L5', 'Jupiter')
#     ]

#     shortest_path = None
#     shortest_distance = np.inf

#     for path in paths:
#         distance = sum(calc_distance(planets[path[i]], planets[path[i+1]]) for i in range(len(path)-1))
#         if distance < shortest_distance:
#             shortest_distance = distance
#             shortest_path = path

#     return shortest_distance, shortest_path

# planets = {}
# for planet in ['Earth', 'Mars', 'Venus', 'Jupiter']:
#     coords = get_input_coords(planet)
#     l4, l5 = calc_lagrange_points(coords)
#     print(f"{planet} L4: {l4}")
#     planets[f'{planet}'] = coords
#     print(f"{planet} L5: {l5}")
#     planets[f'{planet} L4'] = l4
#     planets[f'{planet} L5'] = l5

# shortest_distance, shortest_path = find_shortest_path(planets)
# print(f"The shortest path is {shortest_path} with a distance of {shortest_distance}")

import numpy as np
import itertools
import csv
import matplotlib.pyplot as plt

def read_coordinates_from_csv(filename):
    planets = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            earth_coords = (float(row[0]), float(row[1]))
            venus_coords = (float(row[2]), float(row[3]))
            mars_coords = (float(row[4]), float(row[5]))
            jupiter_coords = (float(row[6]), float(row[7]))

            # Calculate Lagrange points for Earth
            d_earth = np.sqrt(earth_coords[0]**2 + earth_coords[1]**2)
            angle = np.radians(60)
            earth_l4 = earth_coords + d_earth * np.array([np.cos(angle), np.sin(angle)])
            earth_l5 = earth_coords + d_earth * np.array([np.cos(-angle), np.sin(-angle)])

            # Calculate Lagrange points for Venus
            d_venus = np.sqrt(venus_coords[0]**2 + venus_coords[1]**2)
            venus_l4 = venus_coords + d_venus * np.array([np.cos(angle), np.sin(angle)])
            venus_l5 = venus_coords + d_venus * np.array([np.cos(-angle), np.sin(-angle)])

            # Calculate Lagrange points for Mars
            d_mars = np.sqrt(mars_coords[0]**2 + mars_coords[1]**2)
            mars_l4 = mars_coords + d_mars * np.array([np.cos(angle), np.sin(angle)])
            mars_l5 = mars_coords + d_mars * np.array([np.cos(-angle), np.sin(-angle)])

            planets.append({
                'Earth': earth_coords,
                'Venus': venus_coords,
                'Mars': mars_coords,
                'Jupiter': jupiter_coords,
                'Earth L4': earth_l4,
                'Earth L5': earth_l5,
                'Venus L4': venus_l4,
                'Venus L5': venus_l5,
                'Mars L4': mars_l4,
                'Mars L5': mars_l5
            })
    return planets


def calc_lagrange_points(planet_coords):
    d = np.sqrt(planet_coords['Earth'][0]**2 + planet_coords['Earth'][1]**2)
    angle = np.radians(60)
    l4 = planet_coords['Earth'] + d * np.array([np.cos(angle), np.sin(angle)])
    l5 = planet_coords['Earth'] + d * np.array([np.cos(-angle), np.sin(-angle)])
    return l4, l5

def cartesian_to_heliocentric(coords):
    mu = np.sqrt(coords[0]**2 + coords[1]**2)
    nu = np.arctan2(coords[1], coords[0])
    return mu, nu

def calc_distance(coords1, coords2):
    mu1, nu1 = cartesian_to_heliocentric(coords1)
    mu2, nu2 = cartesian_to_heliocentric(coords2)
    return np.sqrt(mu1**2 + mu2**2 - 2*mu1*mu2*np.cos(nu1-nu2))

def find_shortest_path(planets):
    paths = [
        ('Earth', 'Earth L4', 'Mars L4', 'Jupiter'),
        ('Earth', 'Earth L4', 'Mars L5', 'Jupiter'),
        ('Earth', 'Earth L4', 'Venus L4', 'Jupiter'),
        ('Earth', 'Earth L4', 'Venus L5', 'Jupiter'),
        ('Earth', 'Earth L5', 'Mars L4', 'Jupiter'),
        ('Earth', 'Earth L5', 'Mars L5', 'Jupiter'),
        ('Earth', 'Earth L5', 'Venus L4', 'Jupiter'),
        ('Earth', 'Earth L5', 'Venus L5', 'Jupiter'),
    ]

    shortest_path = None
    shortest_distance = np.inf

    for path in paths:
        distance = sum(calc_distance(planets[path[i]], planets[path[i+1]]) for i in range(len(path)-1))
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_path = path

    return shortest_path

def plot_bar_graph(path_counts):
    paths = list(path_counts.keys())
    counts = list(path_counts.values())
    plt.bar(paths, counts)
    plt.xlabel('Shortest Path')
    plt.ylabel('Frequency')
    plt.title('Frequency of Shortest Paths')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Read coordinates from CSV
filename = 'coordinates.csv'
planets_list = read_coordinates_from_csv(filename)

# Find shortest paths for each scenario
path_counts = {}
for idx, planets in enumerate(planets_list, 1):
    shortest_path = find_shortest_path(planets)
    path_str = ' -> '.join(shortest_path)
    if path_str not in path_counts:
        path_counts[path_str] = 0
    path_counts[path_str] += 1

# Plot the bar graph
plot_bar_graph(path_counts)
