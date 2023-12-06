import pandas as pd
import streamlit as st

st.title("Eskwelabs Capstone Project")
st.caption("Description of the app here")
st.image("https://img.freepik.com/free-vector/tiny-female-chef-cooking-vegan-meal-using-recipe-kitchen-cook-making-dish-from-restaurant-menu-flat-vector-illustration-healthy-food-diet-culinary-nutrition-concept-website-design_74855-22063.jpg?w=1800&t=st=1701863073~exp=1701863673~hmac=694591b1fe66e00ba5db832192d8a7a8b4d3f8a3ed3f727bd9cad32131006599")
st.divider()

st.selectbox("Choose recipe:", options=[""])

def function():
	return ""

if st.button("Submit", type="primary"):
	function()

st.divider()