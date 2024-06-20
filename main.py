import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# Rotated Hyper-Ellipsoid Function in 3D
def hyper_ellipsoid_function(x):
    z = 0
    for i in range(len(x)):
        for j in range(i + 1):
            z += x[j]**2
    return z

# Funkcja Michalewicza
def michalewicz_function(x, m=10):
    z = 0
    for i in range(len(x)):
        z += np.sin(x[i]) * (np.sin((i + 1) * x[i]**2 / np.pi))**(2 * m)
    return -z

# Funkcja Styblinski-Tang
def styblinski_tang_function(x):
    x = np.array(x)
    return np.sum(x**4 - 16 * x**2 + 5 * x) / 2.0

# Funkcja Rosenbrock
def rosenbrock_function(x):
    x = np.array(x)
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Funkcja Sferyczna
def sphere_function(x):
    x = np.array(x)
    return np.sum(x**2)

#
# Funkcja aktualizacji prędkości
def update_velocity(particle, velocity, pbest, gbest, w, c=0.1):
    num_particle = len(particle)
    new_velocity = np.zeros(num_particle)
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    c1 = c
    c2 = c
    for i in range(num_particle):
        new_velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i])
    return new_velocity

# Funkcja aktualizacji pozycji
def update_position(particle, velocity, position_min, position_max):
    new_particle = particle + velocity
    new_particle = np.clip(new_particle, position_min, position_max)  # Ensure particles stay within bounds
    return new_particle

# Funkcja PSO
def pso_3d(population, dimension, position_min, position_max, generation, fitness_criterion, fitness_function):
    particles = np.array([[random.uniform(position_min, position_max) for j in range(dimension)] for i in range(population)])
    pbest_position = np.copy(particles)
    pbest_fitness = np.array([fitness_function(p) for p in particles])
    gbest_index = np.argmin(pbest_fitness)
    gbest_position = pbest_position[gbest_index]
    velocity = np.zeros((population, dimension))
    images = []

    early_stop = False
    for t in range(generation):
        if early_stop:
            break

        w = 0.9 - 0.7 * (t / generation)  # Z biegiem czasu zmniejsza masę bezwładności
        for n in range(population):
            velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position, w)
            particles[n] = update_position(particles[n], velocity[n], position_min, position_max)
            fitness_value = fitness_function(particles[n])
            if fitness_value < pbest_fitness[n]:
                pbest_fitness[n] = fitness_value
                pbest_position[n] = particles[n]

        gbest_index = np.argmin(pbest_fitness)
        gbest_position = pbest_position[gbest_index]

        image = ax.scatter([particles[n][0] for n in range(population)],
                           [particles[n][1] for n in range(population)],
                           [fitness_function(particles[n]) for n in range(population)], c='b')
        images.append([image])

        # Warunek wczesnego zatrzymania funkcji hiperelipsoidalnej
        if fitness_function == hyper_ellipsoid_function and np.average(pbest_fitness) <= fitness_criterion:
            early_stop = True

    print('Najlepsza globalna pozycja: ', gbest_position)
    print('Najlepsza wartość dopasowania: ', min(pbest_fitness))
    print('Średnia wartość dopasowania najlepszych cząstek: ', np.average(pbest_fitness))
    print('Liczba generacji: ', t + 1)

    if not images:
        image = ax.scatter([particles[n][0] for n in range(population)],
                           [particles[n][1] for n in range(population)],
                           [fitness_function(particles[n]) for n in range(population)], c='b')
        images.append([image])

    return images

# Główna funkcja do uruchamiania PSO z wybraną przez użytkownika funkcją fitness
def main():
    print("Wybierz funkcję do optymalizacji:")
    print("1. Rotated Hyper-Ellipsoid Function")
    print("2. Michalewicz Function")
    print("3. Styblinski-Tang Function")
    print("4. Rosenbrock Function")
    print("5. Sphere Function")
    choice = int(input("Wybór (1-5): "))

    if choice == 1:
        fitness_function = hyper_ellipsoid_function
        dimension = 6
        fitness_criterion = 1e-4
        generation = 200
    elif choice == 2:
        fitness_function = michalewicz_function
        dimension = 5
        fitness_criterion = 1e-3
        generation = 300
    elif choice == 3:
        fitness_function = styblinski_tang_function
        dimension = 3
        fitness_criterion = 1e-3
        generation = 100
    elif choice == 4:
        fitness_function = rosenbrock_function
        dimension = 3
        fitness_criterion = 1e-3
        generation = 300
    elif choice == 5:
        fitness_function = sphere_function
        dimension = 3
        fitness_criterion = 1e-3
        generation = 100
    else:
        print("Nieprawidłowy wybór!")
        return

    population = 100
    position_min = 0 if choice == 5 else -10.0
    position_max = np.pi if choice == 5 else 10.0

    fig = plt.figure(figsize=(10, 10))
    global ax
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    x = np.linspace(position_min, position_max, 100)
    y = np.linspace(position_min, position_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = [X[i, j], Y[i, j]] + [np.pi/2] * (dimension - 2)
            Z[i, j] = fitness_function(point)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    z_min = np.min(Z)
    z_max = np.max(Z)
    z_margin = (z_max - z_min) * 0.1
    ax.set_zlim(z_min - z_margin, z_max + z_margin)

    images = pso_3d(population, dimension, position_min, position_max, generation, fitness_criterion, fitness_function)

    animated_image = animation.ArtistAnimation(fig, images, interval=100, blit=True, repeat_delay=1000)
    animated_image.save('./pso_complex.gif', writer='pillow')
    plt.show()

if __name__ == "__main__":
    main()
