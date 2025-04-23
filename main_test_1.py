import pandas as pd
import kagglehub
from ast import literal_eval
import numpy as np

from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.snowball import SnowballStemmer
import opendatasets as od

# get dataset from kaggle
# od.download("https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data?select=movies_metadata.csv", force=True)



"""
GET DATA
"""

def svd_func():
   # Load ratings and prepare Surprise dataset
    ratings = pd.read_csv('ratings_small.csv')
    reader  = Reader(rating_scale=(ratings.rating.min(), ratings.rating.max()))
    data    = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)

    # 5-fold CV to evaluate performance
    svd = SVD(random_state=42)
    cross_validate(svd, data, measures=['RMSE','MAE'], cv=5, verbose=True)

    # Train final SVD on all data
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    return svd


def load_data():
  # specify which fields in the movies database to keep 
  fields = ['title','id', 'genres', 'original_language', 'overview', 'tagline', 'release_date','runtime','vote_average','vote_count']
  movies = pd.read_csv('movies_metadata.csv', usecols=fields)
  # drop columns where the id is not a proper number
  movies = movies.drop(movies[(movies['id'] == '1997-08-20') | (movies['id'] == '2012-09-29') | (movies['id'] == '2014-01-01')].index)
  movies['id'] = movies['id'].astype('int')

  credits = pd.read_csv('credits.csv')
  keywords = pd.read_csv('keywords.csv')

  # # get associated imdbId for each movie
  links_small = pd.read_csv('links_small.csv')
  links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

  """
  MERGE TABLES AND PREPROCESS DATA
  """
  # merge movies database with credits and keywords 
  movies = movies.merge(credits, on='id')
  movies = movies.merge(keywords, on='id')


  movies_small = movies[movies['id'].isin(links_small)]
  return movies_small

movies_small = load_data()

# Initialise svd
svd = svd_func()

# 3. Build candidates DataFrame for user_id recommender
candidates = pd.DataFrame()

# get director name from crew data
def extract_director(obj):
  return [i['name'] for i in obj if i['job'] == 'Director']

# get desired feat(ure) from obj
def extract(obj, feat):
  return [i[feat] for i in obj]


def create_metadata(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def preprocess_data(movies_small):
  movies_small['tagline'] = movies_small['tagline'].fillna('')
  movies_small['description'] = movies_small['overview'] + movies_small['tagline']
  movies_small['description'] = movies_small['description'].fillna('')
  movies_small['genres'] = movies_small['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

  """
  EVALUATE STRINGS FROM DATABASE AND TURN STRING DATA INTO LISTS
  """
  movies_small['cast'] = movies_small['cast'].apply(literal_eval)
  movies_small['crew'] = movies_small['crew'].apply(literal_eval)
  movies_small['keywords'] = movies_small['keywords'].apply(literal_eval)

  # create new column for the director of the movie
  movies_small['director'] = movies_small['crew'].apply(extract_director)
  # extract cast member names and limit to 5
  movies_small['cast'] = movies_small['cast'].apply(lambda x: extract(x, 'name')[:5])
  # extract keywords from database
  movies_small['keywords'] = movies_small['keywords'].apply(lambda x: extract(x, 'name'))

  # normalize cast and crew names, lowercase + remove all spaces
  movies_small['cast'] = movies_small['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
  movies_small['director'] = movies_small['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))


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

  movies_small['keywords'] = movies_small['keywords'].apply(
      lambda x: [stemmer.stem(i.replace(" ", "").lower()) for i in x if i in keys_flat]
  )

  """
  COMBINE METADATA AND FEED TO VECTORIZER
  """
  movies_small['metadata'] = movies_small.apply(create_metadata, axis=1)

  count = CountVectorizer(analyzer='word',ngram_range=(1, 2), stop_words='english')
  count_matrix = count.fit_transform(movies_small['metadata'])
  #cosine_sim = cosine_similarity(count_matrix, count_matrix)
  cosine_sim = linear_kernel(count_matrix, count_matrix)

  movies_small = movies_small.reset_index(drop=True)
  indices = pd.Series(movies_small.index, index=movies_small['title']).drop_duplicates()
  return movies_small, cosine_sim, indices


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim, moveies_small, indices):
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



def get_recommendations_hybrid(title, cosine_sim, moveies_small, indices, userId, top_n=50, rec_count=10):

    # 1. Content-based: get index, TMDb id, and internal movieId
    idx = indices[title]

    # 2. Compute similarities and select top_n candidates (skip itself)
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    candidate_idxs = [i for i, _ in sim_scores]

    # 3. Build candidates DataFrame
    candidates = movies_small.iloc[candidate_idxs][['title','vote_count','vote_average','id']].copy()

    # Convert the TMDb 'id' column to numeric, invalid entries become NaN
    movies_small['id'] = pd.to_numeric(movies_small['id'], errors='coerce')

    # Assign a new 'movieId' based on the DataFrame index + 1 (1-based indexing)
    movies_small['movieId'] = movies_small.index + 1

    # Create a simple mapping DataFrame for title ↔ movieId ↔ TMDb id
    # Build helper lookup tables
    id_to_title = movies_small[['title', 'movieId', 'id']].set_index('id')          # TMDb id → title, movieId

    # 4. Collaborative: predict ratings for each candidate
    candidates['est'] = candidates['id'].apply(
        lambda x: svd.predict(userId, id_to_title.loc[x]['movieId']).est
    )

    # 5. Return top recommendations by estimated rating
    return candidates.sort_values('est', ascending=False).head(rec_count)

    

# return list of all movies
def get_movies():
  return movies_small['title'].tolist()

user_id = 5

def get_user_id():
   return user_id

def get_movies_hybrid():
   return candidates['title'].tolist()

if __name__ == "__main__":
  ms = load_data()
  ms, cosine, indices = preprocess_data(ms)
  title = "Frozen"
  recs = get_recommendations(title, cosine, ms, indices)
  print(recs)
  print("REAL TEST\n")
  #print((get_recommendations_hybrid(1, 'Avatar')))
  print((get_recommendations_hybrid('Avatar', cosine, ms, indices, 5)))