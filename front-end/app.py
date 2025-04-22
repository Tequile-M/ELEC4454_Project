import streamlit as st
import requests
from dataclasses import dataclass
import streamlit.components.v1 as components
from streamlit_searchbox import st_searchbox

# --- Define the Movie data class ---
@dataclass
class Movie:
    title: str = "Unknown Title"
    poster_url: str = None
    overview: str = "No overview available."
    year: int = None

def search_movies(searchterm: str) -> list:
    # Search for movies on the Internet Movie Database (IMDb)
    return []

# imdb_id= 'tt0113497'
# movies_imdb = ['tt0113497', 'tt0113228', 'tt0114885', 'tt0113041']

tmdb_ids = ['8844', '15602','31357', '11862']


def fetch_from_tmdb(movie_id):
    # tmdb path to search movie by imdb id
    # url = f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id"
    # poster_url = f"https://api.themoviedb.org/3/movie/{movie_id}/images"
    movie_url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0MzFhOGJjM2E4MjU1MTJjODIwNDkwZDFjYmE5MzVlNiIsIm5iZiI6MTc0NTIxMjU4MC4zMTIsInN1YiI6IjY4MDVkNGE0YzVjODAzNWZiMDhhMDcyMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.CaQ8Fz7c1aTSbzaOjK4eqFsFOqM9oxkPNQ9W303Z5vA"
    }

    response = requests.get(movie_url, headers=headers)
    if response.status_code != 200:
        st.error(f"TMDb API error: {response.status_code}")
        return Movie()

    movie_data = response.json()
    if not movie_data:
        st.warning("No movie found for that ID.")
        return Movie()

    title = movie_data.get("title", "Unknown Title")
    overview = movie_data.get("overview", "No overview available.")
    poster_path = movie_data.get("poster_path")
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

    return Movie(title=title, poster_url=poster_url, overview=overview)


st.set_page_config(layout="wide")

def search_movies(searchterm: str) -> list[str]:
    # Simulate a movie search
    all_titles = ["Inception", "Titanic", "The Matrix", "The Godfather", "Interstellar"]
    return [title for title in all_titles if searchterm.lower() in title.lower()]

selected_value = st_searchbox(
    search_movies,
    placeholder="Search by title",
    key="searchbox",
)
if selected_value:
    st.success(f"You selected: {selected_value}")

    
st.markdown("### 🎬 Recommended movies")

recommendations = []
for movie_id in tmdb_ids:
    recommendations.append(fetch_from_tmdb(movie_id))

# movie = fetch_from_tmdb(imdb_id)
cards_html = ""
for rec in recommendations:
    cards_html += f"""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <img src="{rec.poster_url}" alt="{rec.title}">
                </div>
                <div class="flip-card-back">
                    <div class='card-content'>
                        <h3>{rec.title}</h3>
                        <p>{rec.overview}</p>
                    </div>
                </div>
            </div>
        </div>
    """


# HTML + CSS + JS flip card
components.html(f"""
    <html>
    <head>
    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 20;
            padding: 0;
            font-family: sans-serif;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: center;
            align-items: flex-start;
            max-width: 100%;
            box-sizing: border-box;
        }}
        .flip-card {{
            background-color: transparent;
            width: 300px;
            height: 450px;
            perspective: 1000px;
            cursor: pointer;
        }}

        .flip-card-inner {{
            position: relative;
            width: 100%;
            height: 100%;
            text-align: center;
            transition: transform 0.6s;
            transform-style: preserve-3d;
        }}

        .flip-card:hover .flip-card-inner {{
            transform: rotateY(180deg);
        }}

        .flip-card-front, .flip-card-back {{
            position: absolute;
            width: 100%;
            height: 100%;
            backface-visibility: hidden;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        .flip-card-front {{
            background-color: #000;
        }}

        .flip-card-front img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        .flip-card-back {{
            background-color: #fff;
            color: black;
            transform: rotateY(180deg);
            display: flex;
            flex-direction: column;
        }}
                
        .card-content {{
            padding: 10px;
        }}
    </style>
    </head>
    <body>
        <div class="container">
            {cards_html}
        </div>      
    </body>
    </html>
""", height=1000)