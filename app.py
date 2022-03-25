import numpy as np 
import pandas as pd
import os

# Load Model
import pickle

import streamlit as st

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted >50].index]

csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'knn.pkl')
knn = pickle.load(open(filename, 'rb'))

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10

    movies_check = movies.copy()
    movies_check['title'] = movies_check['title'].apply(lambda x:x.lower())
    movie_list = movies_check[movies_check['title'].str.contains(movie_name.lower())]  

    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"

def predict_note_authentication(X):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Age
        in: query
        type: text
        required: true                                 
    responses:
        200:
            description: The output values
        
    """
   
def main():
    st.title("Movie Recommender")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Movie Recommender ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Title = st.text_input("Movie Title")

    if st.button("Predict"):
        try:
            pred= get_movie_recommendation(Title) 
            if(len(Title)>0):
                st.dataframe(pred)
            else:
                st.text("Write Movie Name")
        except:
            st.text("Error in Movie Name")

    if st.button("About"):
        st.text("Movie Recommender")
        st.text("Final Year Project")

if __name__=='__main__':
    main()
