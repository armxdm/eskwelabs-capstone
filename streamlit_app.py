import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

recipe_df = pd.read_csv('recipe_df_clustered.csv')
options = recipe_df['recipe_name'].to_list()
feature_cols = ['Calories(kcal)', 'Carbohydrates(g)', 'Protein(g)',
               'Fat(g)', 'Saturated Fat(g)', 'Polyunsaturated Fat(g)', 'Sodium(mg)', 'Potassium(mg)',
               'Fiber(g)', 'Sugar(g)', 'Vitamin A(IU)', 'Vitamin C(mg)', 'Calcium(mg)', 'Iron(mg)','Cholesterol(mg)']


st.markdown("<h1><center>Recipe Recommender Engine</center></h1>", unsafe_allow_html = True)
st.image("https://img.freepik.com/free-vector/tiny-female-chef-cooking-vegan-meal-using-recipe-kitchen-cook-making-dish-from-restaurant-menu-flat-vector-illustration-healthy-food-diet-culinary-nutrition-concept-website-design_74855-22063.jpg?w=1800&t=st=1701863073~exp=1701863673~hmac=694591b1fe66e00ba5db832192d8a7a8b4d3f8a3ed3f727bd9cad32131006599")
st.divider()


selectedRecipe = st.selectbox("Choose recipe:", options=options, index=None)

def showNutrientInfo(seed_data):
    seed_data = recipe_df[recipe_df['recipe_name'] == selectedRecipe].iloc[0]
    st.markdown("<b> Nutrition Info</b>", unsafe_allow_html = True)
    st.markdown("<i> Source: " + seed_data['source'] + "</i>", unsafe_allow_html = True)
    st.table(data=seed_data[feature_cols])

def displayNutrientCheckBox():
    columns = st.columns(3)
    with columns[0]:
        calories = st.checkbox("Calories")
        carbohydrates = st.checkbox("Carbohydrates")
        protein = st.checkbox("Protein")
        fat = st.checkbox("Fat")
        saturated_fat = st.checkbox("Saturated Fat")

    with columns[1]:
        polyunsaturated_fat = st.checkbox("Polyunsaturated Fat")
        sodium = st.checkbox("Sodium")
        potassium = st.checkbox("Potassium")
        fiber = st.checkbox("Fiber")
        sugar = st.checkbox("Sugar")

    with columns[2]:
        vitaminA = st.checkbox("Vitamin A")
        vitaminC = st.checkbox("Vitamin C")
        calcium = st.checkbox("Calcium")
        iron = st.checkbox("Iron")
        cholesterol = st.checkbox("Cholesterol")

def setFeatureCols(selected_feature_cols):
    if calories:
        selected_feature_cols.append('Calories(kcal)')
    if carbohydrates:
        selected_feature_cols.append('Carbohydrates(g)')
    if protein:
        selected_feature_cols.append('Protein(g)')
    if fat:
        selected_feature_cols.append('Fat(g)')
    if saturated_fat:
        selected_feature_cols.append('Saturated Fat(g)')
    if polyunsaturated_fat:
        selected_feature_cols.append('Polyunsaturated Fat(g)')
    if sodium:
        selected_feature_cols.append('Sodium(mg)')
    if potassium:
        selected_feature_cols.append('Potassium(mg)')
    if fiber:
        selected_feature_cols.append('Fiber(g)')
    if sugar:
        selected_feature_cols.append('Sugar(g)')
    if vitaminA:
        selected_feature_cols.append('Vitamin A(IU)')
    if vitaminC:
        selected_feature_cols.append('Vitamin C(mg)')
    if calcium:
        selected_feature_cols.append('Calcium(mg)')
    if iron:
        selected_feature_cols.append('Iron(mg)')
    if cholesterol:
        selected_feature_cols.append('Cholesterol(mg)')

def get_cosine_dist(x,y):
    cosine_dist = 1 - cosine_similarity(x.values.reshape(1, -1), y.values.reshape(1, -1)).flatten()[0]
    return cosine_dist

def displayOutput(output, sortBy):
    output.rename(columns= {'recipe_name':'Recipe Name','source':'Source'}, inplace=True)

    if sortBy:
        highlight = lambda slice_of_df: 'background-color: %s' % 'lightyellow'
        output = output.style.applymap(highlight, subset=pd.IndexSlice[:, [sortBy]])

    st.dataframe(output,
        height=750,
        use_container_width = True,
        column_order = ('Recipe Name', 'Calories(kcal)', 'Cholesterol(mg)', 'Sugar(g)', 'Carbohydrates(g)', 'Protein(g)',
            'Fat(g)', 'Saturated Fat(g)', 'Polyunsaturated Fat(g)', 'Sodium(mg)', 'Potassium(mg)',
            'Fiber(g)',  'Vitamin A(IU)', 'Vitamin C(mg)', 'Calcium(mg)', 'Iron(mg)', 'Source'),
        hide_index=True)

if selectedRecipe:
    showNutrientInfo(selectedRecipe)

    st.divider()

    st.markdown("<h4>Get Recipe Recommendations: </h4>", unsafe_allow_html = True)

    with st.expander("Filter by nutrient info: "):
        columns = st.columns(3)
        with columns[0]:
            calories = st.checkbox("Calories")
            carbohydrates = st.checkbox("Carbohydrates")
            protein = st.checkbox("Protein")
            fat = st.checkbox("Fat")
            saturated_fat = st.checkbox("Saturated Fat")

        with columns[1]:
            polyunsaturated_fat = st.checkbox("Polyunsaturated Fat")
            sodium = st.checkbox("Sodium")
            potassium = st.checkbox("Potassium")
            fiber = st.checkbox("Fiber")
            sugar = st.checkbox("Sugar")

        with columns[2]:
            vitaminA = st.checkbox("Vitamin A")
            vitaminC = st.checkbox("Vitamin C")
            calcium = st.checkbox("Calcium")
            iron = st.checkbox("Iron")
            cholesterol = st.checkbox("Cholesterol")

    with st.expander("Sort by: "):
        sortBy = st.radio("", options=["Similarity Score", "Calories(kcal)", "Sugar(g)", "Cholesterol(mg)", 'Carbohydrates(g)', 'Protein(g)',
            'Fat(g)', 'Saturated Fat(g)', 'Polyunsaturated Fat(g)', 'Sodium(mg)', 'Potassium(mg)',
            'Fiber(g)',  'Vitamin A(IU)', 'Vitamin C(mg)', 'Calcium(mg)', 'Iron(mg)'])

    with st.expander("Other Filters"):
        cholesterolRestricted = st.checkbox("Cholesterol Restricted")
        diabetesRestricted = st.checkbox("Diabestes Restricted")
        lessCalories = st.checkbox("Less Calories")
        lessSugar = st.checkbox("Less Sugar")
        lessCholesterol = st.checkbox("Less Cholesterol")

    if st.button("Submit", type="primary"):
        selected_feature_cols = []
        setFeatureCols(selected_feature_cols)
        if len(selected_feature_cols) == 0:
            selected_feature_cols = feature_cols

        seed_data = recipe_df[recipe_df['recipe_name'] == selectedRecipe].iloc[0]
        recipe_df['cosine_similarity_features'] = recipe_df.apply(lambda x: get_cosine_dist(x[selected_feature_cols],seed_data[selected_feature_cols]), axis=1)
        output = recipe_df[(recipe_df['recipe_name']!=seed_data['recipe_name']) & (recipe_df['Cluster']==seed_data['Cluster'])].sort_values('cosine_similarity_features')[:20].reset_index(drop=True)
        if (sortBy != "Similarity Score"):
            output = output.sort_values(sortBy)
        else:
            sortBy = ""

        print(output)
        if cholesterolRestricted:
            output = output[output['cholesterol_restricted'] == False]

        if diabetesRestricted:
            output = output[output['diabetes_restricted'] == False]

        if lessSugar:
            output = output[recipe_df['Sugar(g)'] < seed_data['Sugar(g)']]

        if lessCholesterol:
            output = output[recipe_df['Cholesterol(mg)'] < seed_data['Cholesterol(mg)']]

        if lessCalories:
            output = output[recipe_df['Calories(kcal)'] > seed_data['Calories(kcal)']]

        st.divider()
        st.markdown("<h5>Suggested recipes: </h5>", unsafe_allow_html = True)
        displayOutput(output, sortBy)

        # for i, recipe in output[:5].iterrows():
        #     #st.info(recipe['recipe_name'])
        #     print(str(recipe['cosine_similarity_features']) + "-" + str(recipe['cosine_similarity_features_scaled']))
        #     st.progress(value = recipe['cosine_similarity_features_scaled'])
        #     showNutrientInfo(recipe)
