import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# Fitness function
# Pierwsza funkcja testowa do sprawdzenia działania algorytmu
# f(x1,x2)=(x1+2*-x2+3)^2 + (2*x1+x2-8)^2
# Poszukiwanie minimum ktore wynosi 0

def fitness_function(x1, x2):
    f1 = x1 + 2 * -x2 + 3
    f2 = 2 * x1 + x2 - 8
    z = f1**2 + f2**2
    return z

def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, w_max=1.0, c=0.1):
    # Inicjalizacja nowej tablicy prędkości
    num_particle = len(particle)
    new_velocity = np.zeros(num_particle)
    # Losowe generowanie r1, r2 i wagę bezwładności z rozkładu normalnego
    r1 = random.uniform(0, w_max)
    r2 = random.uniform(0, w_max)
    w = random.uniform(w_min, w_max)
    c1 = c
    c2 = c
    # Obliczanie nowej prędkość
    for i in range(num_particle):
        new_velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i])
    return new_velocity

def update_position(particle, velocity):
    # Przenoszenie czasteczki, dodając prędkość
    new_particle = particle + velocity
    return new_particle

def pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion):
    # Inicjalizacja
    # Populacji
    particles = np.array([[random.uniform(position_min, position_max) for j in range(dimension)] for i in range(population)])
    # Najlepsza pozycja cząstki
    pbest_position = np.copy(particles)
    # Wartość dopasowania
    pbest_fitness = np.array([fitness_function(p[0], p[1]) for p in particles])
    # Indeks najlepszej cząstki
    gbest_index = np.argmin(pbest_fitness)
    # Pozycja najlepszej cząstki globalnej
    gbest_position = pbest_position[gbest_index]
    # Prędkość (rozpoczynając od zerowej prędkości)
    velocity = np.zeros((population, dimension))

    # Miejsce na animację
    images = []

    # Pętla na liczbę generacji
    for t in range(generation):
        # Zatrzymać, jeśli średnia wartość dopasowania osiągnęła zdefiniowane kryterium sukcesu
        if np.average(pbest_fitness) <= fitness_criterion:
            break
        else:
            for n in range(population):
                # Zaktualizowanie prędkości każdej cząstki
                velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)
                # Przenoszenie cząstki na nową pozycję
                particles[n] = update_position(particles[n], velocity[n])
                # Obliczanie wartość dopasowania
                fitness_value = fitness_function(particles[n][0], particles[n][1])
                # Zaktualizowanie najlepszego wyniku
                if fitness_value < pbest_fitness[n]:
                    pbest_fitness[n] = fitness_value
                    pbest_position[n] = particles[n]

        # Znajdowanie indeks najlepszej cząstki
        gbest_index = np.argmin(pbest_fitness)
        # Zaktualizowanie pozycji najlepszej cząstki
        gbest_position = pbest_position[gbest_index]

        # Do animacji: uchwyć stan w każdej generacji
        image = ax.scatter3D([particles[n][0] for n in range(population)],
                             [particles[n][1] for n in range(population)],
                             [fitness_function(particles[n][0], particles[n][1]) for n in range(population)], c='b')
        images.append([image])

    # Wyświetlić wyniki
    print('Najlepsza globalna pozycja: ', gbest_position)
    print('Najlepsza wartość dopasowania: ', min(pbest_fitness))
    print('Średnia wartość dopasowania najlepszych cząstek: ', np.average(pbest_fitness))
    print('Liczba generacji: ', t)

    return images

population = 100
dimension = 2
position_min = -100.0
position_max = 100.0
generation = 400
fitness_criterion = 10e-4

# Przygotowanie do rysowania
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
x = np.linspace(position_min, position_max, 80)
y = np.linspace(position_min, position_max, 80)
X, Y = np.meshgrid(x, y)
Z = fitness_function(X, Y)
ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.2)

# Uruchom PSO i uchwyć klatki animacji
images = pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion)

# Wygenerowanie animacji
animated_image = animation.ArtistAnimation(fig, images)
animated_image.save('./pso_simple.gif', writer='pillow')
plt.show()
