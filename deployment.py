# Import necessary libraries
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Load the model
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load each model (provide the paths to your saved models)
model1 = load_model('model1.pkl')
model2 = load_model('model2.pkl')
model3 = load_model('model3.pkl')
model4 = load_model('model4.pkl')

# Select the model to use
selected_model = st.selectbox('Select a model', ('Model 1', 'Model 2', 'Model 3','model 4)) 

# read data from csv file
data= pd.read_csv("IN_youtube_trending_data.csv")

# Set the App title
st.title("Viral Content Detection in Youtube")

# Adding Inputs

# Define the minimum and maximum years you want to allow in the date input
min_year = datetime.now().year - 20  # Set a minimum year (e.g., 10 years ago)
max_year = datetime.now().year   # Set a maximum year (e.g., 10 years in the future)

# Create a date input with the specified year range
selected_date = st.date_input("Publish Date of the video:", 
                              min_value=datetime(min_year, 1 , 1),
                              max_value=datetime(max_year, 12, 31))


option = st.selectbox(
    'Category of video',
    ('Entertainment', 'Music', 'Film & Animation','Autos & Vehicles','Pets & Animals','Short Movies','Sports','Travel & Events',
     'Gaming','Videoblogging','People & Blogs', 'Comedy', 'Entertainment', 'News & Politics', 'Howto & Style', 'Education', 'Science & Technology', 'Movies', 'Anime/Animation', 'Action/Adventure', 'Classics', 'Comedy', 'Documentary', 'Drama', 'Family', 'Foreign', 'Horror', 'Sci-Fi/Fantasy', 'Thriller', 'Shorts', 'Shows', 'Trailers'
    )
)

if (option=='Film & Animation'):
    option=1
elif (option=='Autos & Vehicles'):
    option=2
elif(option=='Music'):
    option=10

elif(option=='Pets & Animals'):
    option=15

elif(option=='Sports'):
    option=17

elif(option=='Short Movies'):
    option=18

elif(option=='Travel & Events'):
    option=19
elif(option=='Gaming'):
    option=20

elif(option=='Videoblogging'):
    option=21
elif(option=='People & Blogs'):
    option=22
elif(option=='Comedy'):
    option=23
elif(option=='Entertainment'):
    option=24
elif(option=='News & Politics'):
    option=25
elif(option=='Howto & Style'):
    option=26
elif(option=='Education'):
    option=27
elif(option=='Science & Technology'):
    option=28
elif(option=='Movies'):
    option=30
elif(option=='Anime/Animation'):
    option=31
elif(option=='Action/Adventure'):
    option=32
elif(option=='Classics'):
    options=33
elif(option=='Documentry'):
    option=35
elif(option=='Drama'):
    option=36
elif(option=='Family'):
    option=37
elif(option=='Foreign'):
    option=38
elif(option=='Horror'):
    option=39
elif(option=='Sci-Fi/Fantasy'):
    option=40
elif(option=='Thriller'):
    option=41
elif(option=='Shorts'):
    option=42
elif(option=='Shows'):
    option=43
elif(option=='Trailers'):
    option=44



st.write("Input 3: Likes")
Likes=st.number_input("Enter a number of likes")

st.write("Input 4: Dislikes")
Dislikes= st.number_input("Enter a number of dislikes")

st.write("Input 5: Number of Comments")
Comments_Count=st.number_input("Comments count")

selected_option_R = st.radio("Ratings disabled?:", ["Yes", "No"])

selected_option_C= st.radio("Comments disabled?:", ["Yes", "No"])



# Create a prediction button
if st.button('Predict'):
    # Prepare the input features as a numpy array
    input_data = np.array([[selected_date,option,Likes,Dislikes,Comments_Count, selected_option_C, selected_option_R]])

    # Make predictions using the selected model
    if selected_model == 'Model 1':
        prediction = model1.predict(input_data)
    elif selected_model == 'Model 2':
        prediction = model2.predict(input_data)
    elif selected_model == 'Model 3':
        prediction = model3.predict(input_data)
    elif selected_model == 'Model 4':
        prediction = model4.predict(input_data)


    # Display the prediction
    st.write(f'Prediction: {prediction[0]}')

clicked=st.button("Predict")
print(clicked)
# Changes in the code

st.header("Dataset Used: IN_youtube_trending_data.csv")

data= pd.read_csv("IN_youtube_trending_data.csv") # read csv file
st.dataframe(data.head()) # show data in table format

st.write("Data Description:")
st.write(data.describe())




