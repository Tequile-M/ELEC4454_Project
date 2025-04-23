import pandas as pd
import kagglehub
from ast import literal_eval
import numpy as np

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.stem.snowball import SnowballStemmer
import opendatasets as od

# get dataset from kaggle
od.download("https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data?select=movies_metadata.csv", force=True)


# specify which fields in the movies database to keep 
fields = ['title','id', 'genres', 'original_language', 'overview', 'tagline']
movies = pd.read_csv('movies_metadata.csv', usecols=fields)


# drop columns where the id is not a proper number
movies = movies.drop(movies[(movies['id'] == '1997-08-20') | (movies['id'] == '2012-09-29') | (movies['id'] == '2014-01-01')].index)

movies['id'] = movies['id'].astype('int')

credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

# # get associated tmdbId for each movie
# links_small = pd.read_csv('links_small.csv')
# links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# movies['id'] = pd.to_numeric(movies['id'])
# keywords['id'] = keywords['id'].astype('int')
# credits['id'] = credits['id'].astype('int')

# merge movies database with credits and keywords 
movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

print("###############################")


movies_small = movies[movies['id'].isin(links_small)]

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

# movies_small.info()

movies_small['tagline'] = movies_small['tagline'].fillna('')
movies_small['description'] = movies_small['overview'] + movies_small['tagline']
movies_small['description'] = movies_small['description'].fillna('')
movies_small['genres'] = movies_small['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
"""
analyzer='word': analyzes text at the word level; tokenizes the text by words).
ngram_range=(1, 2):captures unigrams (1-word tokens) and bigrams (2-word tokens).
Example: "space battle" → tokens = ["space", "battle", "space battle"]
"""
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(movies_small['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


print("NEWNEWNEWNEWNEWNEWNEWNWENWENWENWENWENWENWENWENWENWENWENWEN")

movies_small['cast'] = movies_small['cast'].apply(literal_eval)
movies_small['crew'] = movies_small['crew'].apply(literal_eval)
movies_small['keywords'] = movies_small['keywords'].apply(literal_eval)
movies_small['cast_size'] = movies_small['cast'].apply(lambda x: len(x))
movies_small['crew_size'] = movies_small['crew'].apply(lambda x: len(x))

# get director name from crew data
def extract_director(obj):
  return [i['name'] for i in literal_eval(obj) if i['job'] == 'Director']

# get desired feat(ure) from obj
def extract(obj, feat):
  return [i[feat] for i in literal_eval(obj)]

movies_small['director'] = movies_small['crew'].apply(extract_director)
# extract cast member names and limit to 5
movies_small['cast'] = movies_small['cast'].apply(lambda x: extract(x, 'name')[:5])
# extract keywords from database
movies_small['keywords'] = movies_small['keywords'].apply(lambda x: extract(x, 'name'))

# normalize cast and crew names, lowercase + remove all spaces
movies_small['cast'] = movies_small['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
movies_small['director'] = movies_small['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
# give more weight to the director than later content-based recs
# movies_small['director'] = movies_small['director'].apply(lambda x: [x,x, x])

print("MAMAMAMAMAMAMAMAMAAMAMAMAMAMAMAAM")

# pre-processing of keywords
# expands keywords then flattens them into a column series
keys_flat = movies_small.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
keys_flat.name = 'keyword'
# count keyword frequency
keys_flat = keys_flat.value_counts()

# remove keywords that only occur once
keys_flat = keys_flat[keys_flat > 1]

# apply stemming (normalize text data for nlp task)
stemmer = SnowballStemmer('english')

# def filter_keywords(x):
#     words = []
#     for i in x:
#         if i in keys_flat:
#             words.append(i)
#     return words

# filter out keywords that only occur once
# movies_small['keywords'] = movies_small['keywords'].apply(lambda x: [i for i in x if i in keys_flat])
# movies_small['keywords'] = movies_small['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
# movies_small['keywords'] = movies_small['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
movies_small['keywords'] = movies_small['keywords'].apply(
    lambda x: [stemmer.stem(i.replace(" ", "").lower()) for i in x if i in keys_flat]
)

movies_small['soup'] = movies_small['keywords'] + movies_small['cast'] + movies_small['director'] + movies_small['genres']

movies_small['soup'] = movies_small['soup'].apply(lambda x: ' '.join(x))

print("EEEEEEEEEEEEEEEEEEEEEEEE")

count = CountVectorizer(analyzer='word',ngram_range=(1, 2), stop_words='english')
count_matrix = count.fit_transform(movies_small['soup'])
#cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim = linear_kernel(count_matrix, count_matrix)


print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")


movies_small = movies_small.reset_index()
titles = movies_small['title']
indices = pd.Series(movies_small.index, index=movies_small['title'])



print("aaaaaaaaaaaaaaaaaaaaaaaaaaa")

# Compute the cosine similarity matrix
#cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


movies_small = movies_small.reset_index(drop=True)

# Then recreate the reverse index mapping
indices = pd.Series(movies_small.index, index=movies_small['title']).drop_duplicates()

#Construct a reverse map of indices and movie titles
indices = pd.Series(movies_small.index, index=movies_small['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_small['title'].iloc[movie_indices]

print(get_recommendations("Harry Potter and the Philosopher's Stone"))


# credits = pd.read_csv('../input/credits.csv')
# keywords = pd.read_csv('keywords.csv')

# keywords['id'] = keywords['id'].astype('int')
# credits['id'] = credits['id'].astype('int')
# md['id'] = md['id'].astype('int')





#NOT MINE

# #Replace NaN with an empty string (uses movies overview for vectorization)
# overview = movies['overview'].fillna('')

# #Construct the required TF-IDF matrix by fitting and transforming the data
# overview_matrix = vectorizer.fit_transform(overview)

# #Output the shape of tfidf_matrix
# overview_matrix.shape



print("\n*********************************************************************************\n")





# credits = pd.read_csv('credits.csv')

# movies['id'] = pd.to_numeric(movies['id'])
# db = movies.merge(credits, on="id")
# db = db.merge(keywords, on="id")

# pd.set_option('display.max_columns', None)
# print(db.info())

# print(db.shape)

# db.iloc[0].genres
# db.iloc[0].crew

# # # Parse the stringified features into their corresponding python objects
# from ast import literal_eval

# def extract(obj, feat):
#   return [i[feat] for i in literal_eval(obj)]

# def extract_director(obj):
#   return [i['name'] for i in literal_eval(obj) if i['job'] == 'Director']

# # Extract 'name' from keywords and genres
# # for col in ['keywords', 'genres']:
# #     db[col] = db[col].apply(lambda x: extract(x, 'name'))

# db['crew'] = db['crew'].apply(extract_director)

# db.iloc[0].crew

# # movies['id'] = pd.to_numeric(movies['id'])

# #get ratingsdatabase
# ratings = pd.read_csv('ratings.csv')

# print(ratings.head())

# print("*********************************************************************************")

# #rate_pivot = ratings.pivot_table(values='rating',columns='userId',index='movieId')


# # plot
# # director
# # crew
# # keywords
