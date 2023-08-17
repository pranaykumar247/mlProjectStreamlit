import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# import seaborn as sns

def main():
    st.set_page_config(page_title = "CAPM",
    page_icon = "chart_with_upwards_trend",
    layout = 'wide')
    st.title("Semi Auto ML App")
    # st.text("Using Streamlit")
    
    
    activities = ["EDA","Plot","Model Building","About"]
    
    choice = st.sidebar.selectbox("Select Activity",activities)
    
    if choice == 'EDA':
        st.subheader("Exploratory Data Analysis")
        
        data = st.file_uploader("Upload Dataset",type=['csv','txt','xlsx'])
        
        if data is not None:
            df = pd.read_csv(data)
            col1,col2 = st.columns([1,1])
            with col1:
                st.dataframe(df.head())
            with col2:
                st.dataframe(df.tail())
            
            if st.checkbox("Show shape"):
                st.write(df.shape)
                
            if st.checkbox("Show columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
            
            if st.checkbox("Select Columns To Show"):
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)          
            if st.checkbox("Show Summary"):
                st.write(df.describe())
                
            if st.checkbox("Show Value Counts"):
                st.write(df.iloc[:,-1].value_counts())
        
    elif choice == 'Plot':
        st.subheader("Data Visualization")
        
        data = st.file_uploader("Upload Dataset",type=['csv','txt','xlsx'])
        
        if data is not None:
            df = pd.read_csv(data)
            col1,col2 = st.columns([1,1])
            with col1:
                st.dataframe(df.head())
            with col2:
                st.dataframe(df.tail())
        
        
        
    elif choice == 'Model Building':
        st.subheader("Models for the dataset")
        
    elif choice == 'About':
        st.subheader("About us")
             
if __name__ == '__main__':
    main()
    
