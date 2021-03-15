from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import six
import sys
from sklearn.metrics import accuracy_score, f1_score

sys.modules['sklearn.externals.six'] = six
import mlrose_hiive
import pandas as pd
import os
import matplotlib.pyplot as plt
from time import time


#Adapted from https://mlrose.readthedocs.io/en/stable/source/tutorial3.html

root_directory = os.path.dirname(__file__)
file_path = fr"{root_directory}\datasets\pd_speech_features\pd_speech_features.csv"
data = pd.read_csv(file_path, ',')
column_y = 'class'
columns_x = [column for column in data.columns if column != column_y]

x = data[columns_x]
y = data[column_y]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Normalize feature data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


algorithms = ['gradient_descent', 'random_hill_climb', 'simulated_annealing', 'genetic_alg']

directory = rf'output\part1'
Path(directory).mkdir(parents=True, exist_ok=True)

for algorithm in algorithms:
    print(algorithm)
    time_start = time()
    nn_model1 = mlrose_hiive.NeuralNetwork(hidden_nodes=[100],
                                           activation='tanh',
                                           algorithm=algorithm,
                                           max_iters=4000,
                                           bias=True,
                                           is_classifier=True,
                                           learning_rate=1,
                                           early_stopping=True,
                                           clip_max=5,
                                           max_attempts=100,
                                           random_state=3,
                                           curve=True)

    nn_model1.fit(X_train_scaled, y_train)

    time_end = time()
    duration = round(time_end - time_start, 5)

    y_train_pred = nn_model1.predict(X_train_scaled)

    y_train_accuracy = accuracy_score(y_train, y_train_pred)

    fitness_curve = nn_model1.fitness_curve

    fig = plt.figure()
    if len(fitness_curve.shape) == 1:
        plt.plot(fitness_curve[1:])
    else:
        plt.plot(fitness_curve[1:, 0])

    fig.suptitle(algorithm)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')

    fig.savefig(rf"{directory}\fitness_curve-{algorithm}.png")
    plt.close()

    print(y_train_accuracy)

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = f1_score(y_test, y_test_pred)

    print(y_test_accuracy)
    print('train time', duration)
