import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

recipe_df = pd.read_csv('recipe_with_ingredients_tfidf.csv')
options = recipe_df['recipe_name'].to_list()
# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Apply the vectorizer to the stringified token lists
tfidf_matrix = tfidf_vectorizer.fit_transform(recipe_df['cleaned_tokenized_ingredients'])

# Initialize NearestNeighbors
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

st.title("Recipe Recommender")
st.caption("Description of the app here")
st.image("https://img.freepik.com/free-vector/tiny-female-chef-cooking-vegan-meal-using-recipe-kitchen-cook-making-dish-from-restaurant-menu-flat-vector-illustration-healthy-food-diet-culinary-nutrition-concept-website-design_74855-22063.jpg?w=1800&t=st=1701863073~exp=1701863673~hmac=694591b1fe66e00ba5db832192d8a7a8b4d3f8a3ed3f727bd9cad32131006599")
st.divider()

selectedRecipe = st.selectbox("Choose recipe:", options=options)

def recommend(dish_name, num_recommendations=5):
    dish_idx = recipe_df.index[recipe_df['recipe_name'] == dish_name].tolist()[0]
    dish_tfidf_vector = tfidf_matrix[dish_idx]
    neighbors = nn.kneighbors(dish_tfidf_vector, n_neighbors=num_recommendations + 1)
    neighbor_indices = neighbors[1][0]
    recommended_dishes = recipe_df['recipe_name'].iloc[neighbor_indices[1:]].tolist()
    return recommended_dishes[:num_recommendations]

if st.button("Submit", type="primary"):
	output = recommend(selectedRecipe)
	st.divider()
	st.markdown("<h5>Suggested recipes: </h5>", unsafe_allow_html = True)
	for val in output:
		st.info(val)

