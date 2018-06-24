import pandas as pd
import os

root  = 'movie_profiles/'
files = [root + fn for fn in os.listdir(root)]

movies_genres = {}


for fn in files:
    
    df = pd.DataFrame.from_csv(fn)
    
    
    for movie, genres in df['genre_tags'].to_dict().items():

        if movie not in movies_genres:
            movies_genres[movie]  = genres.split('\t')
        else:
            movies_genres[movie] += genres.split('\t')
    
    
fout = open('movies_genre_tags.dat', 'w')    
for ind, (movie, genres) in enumerate(movies_genres.items()):
    #if ind == 10: break
    fout.write(movie + '\t' + '\t'.join(genres) + '\n')
fout.close()
