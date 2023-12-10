import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

recipe_df = pd.read_csv('recipe_df_clustered.csv')
options = recipe_df['recipe_name'].to_list()
feature_cols = ['Calories(kcal)', 'Carbohydrates(g)', 'Protein(g)',
               'Fat(g)', 'Saturated Fat(g)', 'Polyunsaturated Fat(g)', 'Sodium(mg)', 'Potassium(mg)',
               'Fiber(g)', 'Sugar(g)', 'Vitamin A(IU)', 'Vitamin C(mg)', 'Calcium(mg)', 'Iron(mg)','Cholesterol(mg)']


st.title("Recipe Recommende Engine")
st.image("https://img.freepik.com/free-vector/tiny-female-chef-cooking-vegan-meal-using-recipe-kitchen-cook-making-dish-from-restaurant-menu-flat-vector-illustration-healthy-food-diet-culinary-nutrition-concept-website-design_74855-22063.jpg?w=1800&t=st=1701863073~exp=1701863673~hmac=694591b1fe66e00ba5db832192d8a7a8b4d3f8a3ed3f727bd9cad32131006599")
st.divider()

selectedRecipe = st.selectbox("Choose recipe:", options=options)

def showNutrientInfo(seed_data):
    st.markdown("<h5>" + seed_data['recipe_name'] + "</h5>", unsafe_allow_html = True)
    st.text("Source: " + seed_data['source'])
    st.table(data=seed_data[feature_cols])

def get_cosine_dist(x,y):
    cosine_dist = 1 - cosine_similarity(x.values.reshape(1, -1), y.values.reshape(1, -1)).flatten()[0]
    return cosine_dist

if st.button("Submit", type="primary"):
    seed_data = recipe_df[recipe_df['recipe_name'] == selectedRecipe].iloc[0]
    recipe_df['cosine_similarity_features'] = recipe_df.apply(lambda x: get_cosine_dist(x[feature_cols],seed_data[feature_cols]), axis=1)
    output = recipe_df[(recipe_df['recipe_name']!=seed_data['recipe_name']) & (recipe_df['Cluster']==seed_data['Cluster'])].sort_values('cosine_similarity_features').reset_index(drop=True)
    st.divider()
    showNutrientInfo(seed_data)

    max_euclidean = recipe_df['cosine_similarity_features'].max() 
    min_euclidean = recipe_df['cosine_similarity_features'].min()
    output['cosine_similarity_features_scaled'] =  (1 - output['cosine_similarity_features'] / (max_euclidean - min_euclidean)).abs()

    st.markdown("<h5>Suggested recipes: </h5>", unsafe_allow_html = True)
    for i, recipe in output[:5].iterrows():
        #st.info(recipe['recipe_name'])
        print(str(recipe['cosine_similarity_features']) + "-" + str(recipe['cosine_similarity_features_scaled']))
        st.progress(value = recipe['cosine_similarity_features_scaled'])
        showNutrientInfo(recipe)
