import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib    

# Load the pre-trained model
pipe_lr = joblib.load(open("model\\text_emotion (1).pkl", "rb"))

# Emoji dictionary for emotions
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮"
}

# Song recommendations based on predicted emotions
song_recommendations = {
    "anger": [
        {"title": "Break Stuff", "artist": "Limp Bizkit", "link": "https://open.spotify.com/track/5cZqsjVs6MevCnAkasbEOX?si=9e3ab6fd9f0d4f72"},
        {"title": "Killing in the Name", "artist": "Rage Against The Machine", "link": "https://open.spotify.com/track/59WN2psjkt1tyaxjspN8fp?si=8c4c2a8c28724d78"},
    ],
    "disgust": [
        {"title": "End of begnning", "artist": "Djo", "link": "https://open.spotify.com/track/3qhlB30KknSejmIvZZLjOD?si=ba0810eec10340ef"},
    ],
    "fear": [
        {"title": "Thriller", "artist": "Michael Jackson", "link": "https://open.spotify.com/track/2LlQb7Uoj1kKyGhlkBf9aC?si=82bc802454e743e2"},
        {"title": "We are champions", "artist": "queen", "link": "https://open.spotify.com/track/1lCRw5FEZ1gPDNPzy1K4zW?si=cdb11fcb5e734364" },
    ],
    "happy": [
        {"title": "Happy", "artist": "Pharrell Williams", "link": "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH?si=3b7f3bcb7dee48c0"},
    ],
    "joy": [
        {"title": "Walking on Sunshine", "artist": "Katrina and the Waves", "link": "https://open.spotify.com/track/05wIrZSwuaVWhcv5FfqeH0?si=ae4b05536e7f409d"},
    ],
    "neutral": [
        {"title": "Weightless", "artist": "Marconi Union", "link": "https://open.spotify.com/track/6kkwzB6hXLIONkEk9JciA6?si=a650eb047bc9489d"},
    ],
    "sad": [
        {"title": "Someone Like You", "artist": "Adele", "link": "https://open.spotify.com/track/3bNv3VuUOKgrf5hu3YcuRo?si=17d9de83162d4006"},
    ],
    "sadness": [
        {"title": "Fix You", "artist": "Coldplay", "link": "https://open.spotify.com/track/7LVHVU3tWfcxj5aiPFEW4Q?si=86d80940dfe749bb"},
    ],
    "shame": [
        {"title": "The Sound of Silence", "artist": "Simon & Garfunkel", "link": "https://open.spotify.com/track/3YfS47QufnLDFA71FUsgCM?si=ccb15cc945364a2c"},
    ],
    "surprise": [
        {"title": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "link": "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS?si=19b6bbb0c7b04ea4"},
    ],
}

# Healing recommendations based on emotions
healing_suggestions = {
    "anger": "Try deep breathing exercises or go for a walk to release tension.",
    "disgust": "Try to focus on positive aspects of the situation. Listening to uplifting music may help.",
    "fear": "Take some time to ground yourself. Writing down your fears and rationalizing them could reduce anxiety.",
    "happy": "Enjoy this moment! Spread the positivity to others around you.",
    "joy": "Cherish your joy and share it with loved ones.",
    "neutral": "A neutral state is great for reflection. Consider engaging in a hobby you enjoy.",
    "sad": "Reach out to someone close or engage in self-care activities like reading or taking a bath.",
    "sadness": "Express your feelings by talking to a friend or writing them down. Music and art can also help.",
    "shame": "Forgive yourself. Remember, everyone makes mistakes. Talking to a trusted friend can be healing.",
    "surprise": "Channel your surprise into excitement by exploring new opportunities or experiences."
}
# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Function to get song recommendations based on predicted emotion
def get_song_recommendations(emotion):
    return song_recommendations.get(emotion, [])

def get_healing_suggestions(emotion):
    return healing_suggestions.get(emotion, "Take some time for self-care and reflection.")


# Main function to handle navigation
def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
    body {
        background-color: #f4f4f4;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #4B4B4B;
        color: white;
    }
    h1 {
        color: #4B4B4B;
        text-align: center;
    }
    .emotion-text {
        font-size: 20px;
        color: #FF6347;
    }
    .confidence {
        font-weight: bold;
        font-size: 18px;
        color: #4682B4;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    menu = ["Survey","Home", "Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Page
    if choice == "Home":
        st.markdown("<h1>Welcome to the Text Emotion Detection Web App</h1>", unsafe_allow_html=True)
        st.write("""
        <p style='text-align: center;'>This app uses a machine learning model to detect emotions from text input.
        Use the Emotion Detection page to analyze the emotions in your text.
        Learn more about the project on the About page.</p>
        """, unsafe_allow_html=True)
        st.image("pixlr-image-generator-2d0afb64-9a18-44b3-8297-29d1c4735e73.png", use_column_width=True)

    # Emotion Detection Page
    elif choice == "Emotion Detection":
        st.title("Text Emotion Detection")
        st.subheader("Detect emotions from your text")

        # Add a suggestion box for emotions
        emotion_suggestion = st.selectbox("Select a suggested emotion (optional)", [
            "anger", "disgust", "fear", "happy", "joy", "neutral", 
            "sad", "sadness", "shame", "surprise"
        ])

        with st.form(key='emotion_form'):
            raw_text = st.text_area("Type your text here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(f"<div class='emotion-text'>{raw_text}</div>", unsafe_allow_html=True)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"<div class='emotion-text'>{prediction}: {emoji_icon}</div>", unsafe_allow_html=True)
                st.write(f"<div class='confidence'>Confidence: {np.max(probability):.2f}</div>", unsafe_allow_html=True)

                # Song Recommendations
                st.success("Recommended Songs")
                recommendations = get_song_recommendations(prediction)
                for song in recommendations:
                    st.write(f"- {song['title']} by {song['artist']} ([Listen on Spotify]({song['link']}))")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                # Create a bar chart of prediction probabilities
                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)
                
                
                # Survey Page
    if choice == "Survey":
        st.title("Survey: How Do You Feel Today?")
        st.write("We would love to know how you're feeling. Please fill out this short survey.")

        with st.form(key='survey_form'):
            current_mood = st.selectbox("How are you feeling right now?", list(healing_suggestions.keys()))
            thoughts = st.text_area("What are your thoughts today?")
            submit_survey = st.form_submit_button(label='Submit Survey')

        if submit_survey:
            st.success(f"Thank you for sharing! You feel {current_mood}.")
            st.write(healing_suggestions[current_mood])

   # About Page
     # About Page
    elif choice == "About":
        st.markdown("<h1>About This App</h1>", unsafe_allow_html=True)
        st.write("""
        <p style='text-align: center;'>This app is a text emotion detection tool built using machine learning techniques.
        The model analyzes input text and predicts the emotion, along with the confidence score for each emotion.
        It uses a pre-trained Stochastic Gradient Descent (SGD) classifier with a pipeline of text processing steps.
        <br><br>
        <b>Developed by Sneha Kashitkar.</b></p>
        """, unsafe_allow_html=True)
        
        st.write("""
        <h2>Research on Text Emotion Detection</h2>
        <p>This research focuses on using natural language processing (NLP) techniques to identify and categorize emotions expressed in text. 
        The importance of understanding emotional nuances in textual data is crucial in various applications such as customer feedback analysis, 
        social media sentiment analysis, and mental health monitoring. 
        The model has been trained on a diverse dataset to ensure accuracy and reliability in emotion detection.</p>
        """, unsafe_allow_html=True)
        
        # Add an image related to the project
        st.image("pixlr-image-generator-de7477f1-0ebc-4baa-8806-a37bbd303772.png", use_column_width=True)  # Replace with the correct path to your image
        
        st.write("""
        <h3>License</h3>
        <p>This project is licensed under the MIT License. See the <a href="https://opensource.org/licenses/MIT" target="_blank">LICENSE</a> file for details.</p>
        """, unsafe_allow_html=True)
        
        # Add copyright information
        st.write("<p>© 2024 Sneha Kashitkar. All rights reserved.</p>", unsafe_allow_html=True)

# Running the main function
if __name__ == '__main__':
    main()
