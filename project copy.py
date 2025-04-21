import pandas as pd
import kagglehub
from ast import literal_eval
import numpy as np

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.stem.snowball import SnowballStemmer


print("TESTSSSSSSSSSSSSSS")



# #specify which fields in the movies database to keep (we don't need all of them)
# fields = ['title','id', 'genres', 'original_language', 'overview']


# movies = pd.read_csv('movies_metadata.csv', usecols=fields)
movies = pd.read_csv('movies_metadata.csv')


# drop columns where the id is not a proper number
movies = movies.drop(movies[(movies['id'] == '1997-08-20') | (movies['id'] == '2012-09-29') | (movies['id'] == '2014-01-01')].index)

movies['id'] = movies['id'].astype('int')
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

movies.shape

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')

movies = movies.merge(credits, on='id')
movies = movies.merge(keywords, on='id')

print("###############################")

smovies = movies[movies['id'].isin(links_small)]
smovies.shape

print(smovies.head(5))

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
# movies= keywords.merge(keywords,on='id')
# print("###############################")

smovies.info()

smovies['tagline'] = smovies['tagline'].fillna('')
smovies['description'] = smovies['overview'] + smovies['tagline']
smovies['description'] = smovies['description'].fillna('')
smovies['genres'] = smovies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
"""
analyzer='word':It analyzes text at the word level, meaning it tokenizes the text by words (as opposed to characters).
ngram_range=(1, 2):This captures unigrams (1-word tokens) and bigrams (2-word tokens).
Example: "space battle" → tokens = ["space", "battle", "space battle"]
This helps capture context like "New York" or "space station".
"""
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')

#Replace NaN with an empty string
#smovies['overview'] = smovies['overview'].fillna('')


#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(smovies['description'])

#Output the shape of tfidf_matrix
tfidf_matrix.shape


print("NEWNEWNEWNEWNEWNEWNEWNWENWENWENWENWENWENWENWENWENWENWENWEN")

smovies['cast'] = smovies['cast'].apply(literal_eval)
smovies['crew'] = smovies['crew'].apply(literal_eval)
smovies['keywords'] = smovies['keywords'].apply(literal_eval)
smovies['cast_size'] = smovies['cast'].apply(lambda x: len(x))
smovies['crew_size'] = smovies['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

smovies['director'] = smovies['crew'].apply(get_director)
smovies['cast'] = smovies['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smovies['cast'] = smovies['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smovies['keywords'] = smovies['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

smovies['cast'] = smovies['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smovies['director'] = smovies['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smovies['director'] = smovies['director'].apply(lambda x: [x,x, x])

print("MAMAMAMAMAMAMAMAMAAMAMAMAMAMAMAAM")

s = smovies.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s[:5]

s = s[s > 1]

# Stemming is a text preprocessing technique in natural language processing (NLP).
# Specifically, it is the process of reducing inflected form of a word to one
# so-called “stem,” or root form, also known as a “lemma” in linguistics.
stemmer = SnowballStemmer('english')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words
smovies['keywords'] = smovies['keywords'].apply(filter_keywords)
smovies['keywords'] = smovies['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smovies['keywords'] = smovies['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

smovies['soup'] = smovies['keywords'] + smovies['cast'] + smovies['director'] + smovies['genres']

smovies['soup'] = smovies['soup'].apply(lambda x: ' '.join(x))

print("EEEEEEEEEEEEEEEEEEEEEEEE")

count = CountVectorizer(analyzer='word',ngram_range=(1, 2), stop_words='english')
count_matrix = count.fit_transform(smovies['soup'])
#cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim = linear_kernel(count_matrix, count_matrix)


print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")


smovies = smovies.reset_index()
titles = smovies['title']
indices = pd.Series(smovies.index, index=smovies['title'])



print("aaaaaaaaaaaaaaaaaaaaaaaaaaa")

# Compute the cosine similarity matrix
#cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


smovies = smovies.reset_index(drop=True)

# Then recreate the reverse index mapping
indices = pd.Series(smovies.index, index=smovies['title']).drop_duplicates()

#Construct a reverse map of indices and movie titles
indices = pd.Series(smovies.index, index=smovies['title']).drop_duplicates()

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
    return smovies['title'].iloc[movie_indices]

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
