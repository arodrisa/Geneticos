# %%
# Libraries
from urllib.request import Request, urlopen
import urllib
from bs4 import BeautifulSoup as bs
from pandas.tseries.offsets import BDay
from functools import partial
import itertools
import multiprocessing as mp
import requests
import datetime
import os
import investpy
import re
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from random import sample
from tqdm import tqdm, tqdm_notebook
import random
from random import shuffle
import math

# import sys
# import seaborn as sns

# %%
# Variables
n_ciudades = 47
input_file_path = r"C:\Users\arodr\Google Drive\Master_MIAX\Modulo2\Algoritmos genéticos, enjambre y lógica difusa\cities_distances.csv"
puntos_de_corte = 2


def main_traveller(n_ciudades, input_file_path, puntos_de_corte):

    convergencia = 0

    ciudades_df = read_input_csv(input_file_path)
    [viajes_realizados_df, distancias_list] = generar_poblacion_inicial(
        n_ciudades, ciudades_df[:])
    best_trip = distancias_list[mejor_viajante(distancias_list)]
    counter = 0
    while convergencia <= 200:
        prev_best_trip = best_trip
        viajes_realizados_df = seleccion_padres_y_cruce(
            viajes_realizados_df[:], distancias_list, ciudades_df[:])
        viajes_realizados_df = mutacion_sustitucion(
            viajes_realizados_df[:], ciudades_df[:], n_ciudades)
        distancias_list = []
        for viaje in range(0, len(viajes_realizados_df.columns)):
            distancias_list.append(calcula_distancia(
                viajes_realizados_df.iloc[:, viaje].tolist(), ciudades_df[:]))
        best_trip = distancias_list[mejor_viajante(distancias_list)]
        if(best_trip == prev_best_trip):
            convergencia += 1
        else:
            convergencia = 0
        counter += 1
        if(counter % 50 == 0):

            print(
                f'El mejor viaje por ahora es de {best_trip} km,iteracion {counter}')

    print(f'El mejor viaje es de {best_trip} km')


def read_input_csv(path):
    pd_cities = pd.read_csv(path, encoding='utf-8', index_col=0, sep=";")
    pd_cities = pd_cities.apply(lambda x: x.str.replace('km', ''))
    pd_cities = pd_cities.apply(lambda x: x.str.replace(',', ''))
    pd_cities = pd_cities.apply(pd.to_numeric, errors='coerce')
    return pd_cities


def generar_poblacion_inicial(n_ciudades, ciudades_df):
    bool_firsttime = True
    distancia_list = []

    n_cadenas = n_ciudades * 50 / (5 + n_ciudades)

    for cadena in range(0, math.ceil(n_cadenas)):
        viaje_list = realiza_un_viaje(ciudades_df[:], n_ciudades)
        distancia = calcula_distancia(viaje_list, ciudades_df[:])
        if(bool_firsttime):
            bool_firsttime = False
            cadenas_df = pd.DataFrame(viaje_list, columns=[cadena])
            distancia_list.append(distancia)
        else:
            distancia_list.append(distancia)
            cadenas_df[cadena] = viaje_list
    return[cadenas_df, distancia_list]


def calcula_distancia(list_ciudades, pd_cities):
    total_dist = 0
    for index in range(len(list_ciudades)-1):
        total_dist += pd_cities.loc[list_ciudades[index],
                                    list_ciudades[index+1]]
    return total_dist


def realiza_un_viaje(pd_cities, num_ciudades):
    cities_list = []
    temp_cities_list = pd_cities.index[1:num_ciudades].tolist()
    shuffle(temp_cities_list)
    cities_list.append(pd_cities.index[0])
    cities_list.extend(temp_cities_list)
    cities_list.append(pd_cities.index[0])
    return cities_list


def mejor_viajante(distancias_list):
    return distancias_list.index(min(distancias_list))


def seleccion_padres_y_cruce(viajes_df, distancias_list, ciudades_df):
    mejor_viajante_index = mejor_viajante(distancias_list)
    nuevos_viajantes = pd.DataFrame(viajes_df.iloc[:, mejor_viajante_index])
    mejor_dist = distancias_list[mejor_viajante_index]
    nuevos_padres = []

    # 'Para realizar la selección de los padres de la generación siguiente se establece la siguiente relación, que será la probabilidad de ocurrencia: =2-(distancia recorrida por cromosoma/distancia recorrida por el mejor cromosoma).
    # 'El mejor cromosoma tendrá una probabilidad de ocurrencia del 100%, mientras que el resto tendrá una oportunidad de ocurrencia menor (entre el 0% y el 100%).
    # 'Por ejemplo, Si la probabilidad es un 60% de ocurrencia, se saca un número aleatorio y, Sí es menor al 60% se seleccionará para crear un "hijo" y Sí es mayor no saldrá seleccionado.
    # 'Este proceso se repetirá hasta que se complete la matriz de descendencia.
    # Para elegir qué cromosomas evaluamos para ver Sí van a tener descendencia o no podemos hacer dos cosas: O bien los recorremos en orden evaluándolos. O bien, sacamos los cromosomas aleatoriamente y los evaluamos.
    # 'Sí lo hacemos en órden, los primeros cromosomas tienen más probabilidad que los últimos de salir elegidos, dado que se evuarán, probablemente, más veces. Por lo que nos decantamos por una selección aleatoria.
    while(len(nuevos_padres) != len(viajes_df.columns)):
        cromosoma_random = random.choice(viajes_df.columns)
        if(random.random() < (2-(distancias_list[cromosoma_random]/mejor_dist))):
            nuevos_padres.append(cromosoma_random)

    # 'Una vez seleccionados los padres sacamos los puntos de corte para el cruce.
    # 'La primera y última ciudad debe seguir siendo Amsterdam, por lo que los puntos de cruce no pueden incluirlas.
    # 'El primer viajante es el mejor de la generación anterior y no debe modificarse.
    # 'Podemos usar 1 o 2 puntos de corte.

    for padre in nuevos_padres:
        if puntos_de_corte == 2:
            punto_corte_1 = random.randrange(1, n_ciudades-1)
            punto_corte_2 = random.randrange(punto_corte_1, n_ciudades)
        if puntos_de_corte == 1:
            if rand < 0.5:
                punto_corte_1 = 1
                punto_corte_2 = random.randrange(punto_corte_1, n_ciudades)
            else:
                punto_corte_1 = random.randrange(1, n_ciudades-1)
                punto_corte_2 = n_ciudades-1
        nuevo_viajante = viajes_df.iloc[:, padre]
        nuevo_viajante_list = nuevo_viajante.iloc[punto_corte_1:
                                                  punto_corte_2].tolist()
        shuffle(nuevo_viajante_list)
        nuevo_viajante.iloc[punto_corte_1:punto_corte_2] = nuevo_viajante_list
        nuevos_viajantes[len(nuevos_viajantes.columns)] = nuevo_viajante

        nuevos_viajantes.columns = range(0, len(nuevos_viajantes.columns))
    return nuevos_viajantes


def mutacion_sustitucion(nuevos_viajantes, ciudades_df, n_ciudades):
    mutation_prob = random.random()/10
    for viajante in range(1, len(nuevos_viajantes.columns)):
        if(random.random() < mutation_prob):
            nuevos_viajantes.iloc[:, viajante] = realiza_un_viaje(
                ciudades_df[:], n_ciudades)

    nuevos_viajantes.columns = range(0, len(nuevos_viajantes.columns))
    return nuevos_viajantes


# %%
main_traveller(n_ciudades, input_file_path, puntos_de_corte)

# %%
