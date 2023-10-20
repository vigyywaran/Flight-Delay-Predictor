import streamlit as st
from Input import show_explore_page
def show_readme():
    st.write ("## ReadMe:")
    st.write ("#### This is a attempt to predict the flight delay based on the 2008 flight delay dataset.")
    #st.write ("### This uses Decision Trees to learn and predict the delay in flights (in minutes)")
    st.write ("#### It includes some information (EDA) on the 2008 flight delay data.")
    st.write("#### Please scroll down, input numerical values as asked and click on the Calculate Delay button.")
    st.write ("#### Note : This uses Decision Trees to learn and predict the delay in flights (in minutes)")