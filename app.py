## Framework
import streamlit as st

## EDA
import pandas as pd
import numpy as np

## Visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

## Model Building
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 

def main():
    st.set_page_config(page_title = "CAPM",
    page_icon = "chart_with_upwards_trend",
    layout = 'wide')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    css_style = """
<style>
    body{
        background-color: white;
    }
    .app-title {
        font-size: 45px;
        font-weight: bold;
        color: darkgrey;
        margin-top: 30px;
        margin-bottom: 20px;
        text-align: center;
    }
    .subheader {
        font-size: 30px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
        color: Orange
    }
    .connect {
        font-size: 18px;
        margin-top: 10px;
    }
    .link {
        color: blue;
        text-decoration: none;
    }
    .link:hover {
        color: red;
        text-decoration: underline;
    }
</style>
"""
    st.markdown(css_style, unsafe_allow_html=True)  # Apply inline CSS
    
    st.markdown(
            "<div class='app-title'>Semi Auto ML App</div>",
            unsafe_allow_html=True
        )
    # st.text("Using Streamlit")
    
    activities = ["EDA","Plot","Model Building","About"]
    
    choice = st.sidebar.selectbox("Select Activity",activities)
    
    if choice == 'EDA':
        st.markdown("<div class='subheader'>Exploratory Data Analysis</div>",unsafe_allow_html=True)
        
        data = st.file_uploader("Upload Dataset",type=['csv','txt','xlsx'])
        
        if data is not None:
            encodings = ['utf-8', 'latin1', 'ISO-8859-1']  # Add more encodings if needed
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(data, encoding=encoding)
                    break  # Break if the file is successfully read
                except UnicodeDecodeError:
                    continue  # Try the next encoding if this one fails
            
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
                
            if st.checkbox("Data Types"):
                st.write(df.dtypes)
        
    elif choice == 'Plot':
        # st.subheader("Data Visualization")
        st.markdown("<div class='subheader'>Data Visualization</div>",unsafe_allow_html=True)
        
        data = st.file_uploader("Upload Dataset",type=['csv','txt','xlsx'])
        
        if data is not None:
            encodings = ['utf-8', 'latin1', 'ISO-8859-1']  # Add more encodings if needed
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(data, encoding=encoding)
                    break  # Break if the file is successfully read
                except UnicodeDecodeError:
                    continue  # Try the next encoding if this one fails
            
            
            col1,col2 = st.columns([1,1])
            with col1:
                st.dataframe(df.head())
            with col2:
                st.dataframe(df.tail())
                
            if st.checkbox("Correlation with Seaborn"):
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                corr_df = df[numeric_columns].corr()
                st.write(sns.heatmap(corr_df, annot=True))
                st.pyplot()
                
            if st.checkbox("Pie Chart"):
                all_columns = df.columns.to_list()
                columns_to_plot = st.selectbox('Select 1 column',all_columns)
                pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pie_plot)
                st.pyplot()
                
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot", ["area","bar","line","hist","box","kde"])
            selected_columns_names = st.multiselect("Select Columns To Plot", all_columns_names)
            
            if st.button("Generate Plot"):
                st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))
                
                # Plot by Streamlit
                if type_of_plot == "area":
                    cust_data = df[selected_columns_names]
                    
                    # Check if the selected columns contain categorical data
                    categorical_columns = cust_data.select_dtypes(include=['object']).columns.tolist()
                    
                    if not categorical_columns:
                        # If there are no categorical columns, proceed to create the bar chart
                        st.area_chart(cust_data)
                    else:
                        st.error("Selected columns contain categorical data. Cannot create area chart.")

                    
                elif type_of_plot == "bar":
                    cust_data = df[selected_columns_names]
                    
                    # Check if the selected columns contain categorical data
                    categorical_columns = cust_data.select_dtypes(include=['object']).columns.tolist()
                    
                    if not categorical_columns:
                        # If there are no categorical columns, proceed to create the bar chart
                        st.bar_chart(cust_data)
                    else:
                        st.error("Selected columns contain categorical data. Cannot create bar chart.")

                    
                elif type_of_plot == "line":
                    cust_data = df[selected_columns_names]
                                        # Check if the selected columns contain categorical data
                    categorical_columns = cust_data.select_dtypes(include=['object']).columns.tolist()
                    
                    if not categorical_columns:
                        # If there are no categorical columns, proceed to create the bar chart
                        st.line_chart(cust_data)
                    else:
                        st.error("Selected columns contain categorical data. Cannot create line chart.")
                    
                # Custom Plot
                elif type_of_plot:
                    cust_plot = df[selected_columns_names].plot(kind = type_of_plot)
                    st.write(cust_plot)
                    st.pylot()
                
        
        
    elif choice == 'Model Building':
        # st.subheader("Models for the dataset")
        st.markdown("<div class='subheader'>Models for the dataset</div>",unsafe_allow_html=True)
        
        data = st.file_uploader("Upload Dataset",type=['csv','txt','xlsx'])
        
        if data is not None:
            encodings = ['utf-8', 'latin1', 'ISO-8859-1']  # Add more encodings if needed
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(data, encoding=encoding)
                    break  # Break if the file is successfully read
                except UnicodeDecodeError:
                    continue  # Try the next encoding if this one fails
            
            col1,col2 = st.columns([1,1])
            with col1:
                st.dataframe(df.head())
            with col2:
                st.dataframe(df.tail())
                
            # Select only numerical columns
            # numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
            # df_numerical = df[numerical_columns]
            
            X = df.iloc[:, :-1]
            Y = df.iloc[:, -1] 
            seed = 7
           
            # Model
            models = []
            models.append(("LR",LogisticRegression()))
            models.append(("LDA",LinearDiscriminantAnalysis()))
            models.append(("KNN",KNeighborsClassifier()))
            models.append(("CART",DecisionTreeClassifier()))
            models.append(("NB",GaussianNB()))
            models.append(("SVM",SVC()))
            
            # Evaluating each model 
            ## List
            model_names = []
            model_mean = []
            model_std = []
            all_models = []
            scoring = 'accuracy'
            
            for name,model in models:
                kfold = model_selection.KFold(n_splits=10,shuffle=True,random_state = seed)
                cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold, scoring = scoring)
                model_names.append(name)
                model_mean.append(cv_results.mean())
                model_std.append(cv_results.std())
                
                accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"standard_deviation":cv_results.std()}
                
                all_models.append(accuracy_results)
                
            if st.checkbox("Metrics as Table"):
                st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns=["Model Names","Mean","Std"]))
                
            if st.checkbox("Metrics as JSON"):
                st.json(all_models)
              
        
    elif choice == 'About':
        # st.subheader("Connect with Us")       
        # st.subheader("Project Information")
        st.markdown("<div class='subheader'>Project Information</div>",unsafe_allow_html=True)
        st.markdown(
            "This project was created using Streamlit to semi-automate ML models and EDA practice."
        )
        
        st.subheader("Connect with Us")
        st.markdown(
            "<div class='connect'>Follow us on GitHub: <a class='link' href='https://github.com/pranaykumar247'>GitHub</a></div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<div class='connect'>Connect with us on LinkedIn: <a class='link' href='https://www.linkedin.com/in/pranay-kumar-85862a1bb/'>LinkedIn</a></div>",
            unsafe_allow_html=True
        )
             
if __name__ == '__main__':
    main()
    
