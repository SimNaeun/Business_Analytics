import pandas as pd
movie_full_data = pd.read_csv('tmdb_5000_movies.csv')
movie = movie_full_data[['original_title', 'genres']]

from ast import literal_eval
for i in range(len(movie)):
    mv = movie.iloc[i]
    mv['genres'] = literal_eval(mv['genres'])
    mv['genres'] = [y['name'] for y in mv['genres']]

all_genres = []
for i in range(len(movie)):
    mv = movie.iloc[i]
    for g in mv['genres']: 
        if g not in all_genres: all_genres.append(g)
print(all_genres)

movie = movie.assign(gvec = "")

import numpy as np
for i in range(len(movie)):
    mv = movie.iloc[i]
    vec = []
    for g in all_genres:
        if g in mv['genres']: vec.append(1)
        else: vec.append(0)
    movie.loc[i, 'gvec'] = np.asarray(vec, dtype='float32')

def magnitude(v):
    m=0
    for x in v: m += x**2
    return (np.sqrt(m))

def similarity(a,b):
    dot = np.dot(a,b)
    if dot==0: return 0
    ma = magnitude(a)
    mb = magnitude(b)
    return(dot/(ma*mb))

while True:
    movie_num_str = input("\nEnter movie number: ")
    if movie_num_str == '': break
    movie_num = int(movie_num_str)
    print("You selected: ", movie.loc[movie_num, 'original_title'])
    my_vec = movie.loc[movie_num, 'gvec']
    sim_list = []
    for i in range(len(movie)):
        if i==movie_num: sim_list.append(0)
        else:
            mv = movie.iloc[i]
            sim = similarity(my_vec, mv['gvec'])
            sim_list.append(sim)
    y = np.argsort(sim_list)[::-1]
    print("Recommended:")
    for i in y[0:10]:
        print('%4d'%i, '%6.4f'%sim_list[i], movie.loc[i,'original_title'])

# ex15-1.py
