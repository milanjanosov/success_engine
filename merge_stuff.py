
import os

root = 'ProcessedDataNormalizedRandomized/4_NN_rank_N'

fff = open(root + '/imdb_average_ratings.dat', 'w')
fff.close()
ggg = open(root + '/imdb_rating_counts.dat', 'w')
ggg.close()
hhh = open(root + '/imdb_metascores.dat', 'w')
hhh.close()
iii = open(root + '/imdb_critic_reviews.dat', 'w')
iii.close()
jjj = open(root + '/imdb_user_reviews.dat', 'w')
jjj.close()
rrr = open(root + '/imdb_gross.dat', 'w')
rrr.close()


def save_what(kw, infile, outfile):

    if kw in infile:
        for line in open(root + '/' + infile):
            fff = open(outfile, 'a')             
            fff.write(line)
            fff.close()


for filename in os.listdir(root):

    if 'all' in filename and 'film' in filename:

        save_what('avg_rat', filename, root + '/imdb_average_ratings.dat')
        save_what('rating_count', filename, root + '/imdb_rating_counts.dat')
        save_what('meta', filename, root + '/imdb_metascores.dat')        
        save_what('critic', filename, root + '/imdb_critic_reviews.dat')
        save_what('user', filename, root + '/imdb_user_reviews.dat')
        save_what('gross', filename, root + '/imdb_gross.dat')
