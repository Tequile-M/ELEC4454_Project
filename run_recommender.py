"""
MOVIE RECOMMENDATION ENGINE

"""

import pickle
import pandas as pd

# Load everything from disk
with open("movies_small.pkl", "rb") as f:
    movies_small = pickle.load(f)
with open("cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)
with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)
with open("svd_model.pkl", "rb") as f:
    svd = pickle.load(f)

# Hybrid recommendation using svd and cosine_sim
# cosine_sim is a matrix that tells how similar any two movies are based on content-based filtering
# indices is a dictionary that connects movie titles to their row number in the dataset
# svd: object that holds trained recommendation model
def get_recommendations_hybrid(title, userId, top_n=50, rec_count=10):

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


def get_rec_ids(title):
    userId = 1 #arbitrary
    top_n=50 # how many results to return for content-based filter
    no_recs=10 # number of recs
    recs = get_recommendations_hybrid(title, userId)
    return recs['id'].tolist()
# # Run test
# if __name__ == "__main__":
#     print(get_recommendations("The Matrix", cosine_sim, movies_small, indices))


