import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import streamlit as st
import speech_recognition as sr
import pyaudio..

# Function to load dataset
@st.cache_data
def load_data():
    """
    Loads the dataset from a CSV file.
    Returns a DataFrame or None if there's an error.
    """
    data_path = "data/dataset.csv"
    try:
        df = pd.read_csv(data_path, on_bad_lines='skip')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# Function to train the AI model
@st.cache_resource
def train_model(df):
    """
    Trains a text classification model using TF-IDF and Multinomial Naive Bayes.
    Returns the trained model pipeline.
    """
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['question'], df['answer'])
    return model

# Function to add new entries to the dataset
def add_to_dataset(question, answer, image, video, dataset_path="data/dataset.csv"):
    """
    Adds a new Q&A entry to the dataset.
    """
    new_entry = pd.DataFrame([[question, answer, image, video]], columns=["question", "answer", "image", "video"])
    new_entry.to_csv(dataset_path, mode="a", header=False, index=False)

# Function to check if a microphone is available
def is_microphone_on():
    """
    Checks if any microphone is available for use.
    Returns True if a microphone is detected, False otherwise.
    """
    try:
        pa = pyaudio.PyAudio()
        device_count = pa.get_device_count()
        for i in range(device_count):
            device_info = pa.get_device_info_by_index(i)
            if device_info["maxInputChannels"] > 0:  # Device has input channels
                return True
        return False
    except Exception as e:
        st.error(f"Microphone detection error: {e}")
        return False

# Function to convert speech to text
def speech_to_text():
    """
    Captures speech input using the microphone and converts it to text.
    Returns the recognized text or an empty string if an error occurs.
    """
    if not is_microphone_on():
        st.warning("No active microphone found. Please ensure your microphone is on and try again.")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            # Adjusting recognizer sensitivity for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)  # More flexible timing
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
    except sr.WaitTimeoutError:
        st.error("Listening timed out. Please try speaking again.")
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand that. Please speak clearly.")
    except sr.RequestError as e:
        st.error(f"Error with the speech recognition service: {e}")
    return ""

# Main Streamlit app function
def main():
    st.title("🌟 AI Tutor with Multimedia Responses")
    st.subheader("Ask me any question about AI, and I can respond with images or videos!")

    # Load dataset
    df = load_data()
    if df is None or df.empty:
        st.error("Dataset failed to load. Please check the file format and try again.")
        return

    # Train model
    model = train_model(df)

    # Initialize conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Input options: Text or Voice
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_input("Type your question here:")
    with col2:
        if st.button("Use Voice"):
            user_input = speech_to_text()

    # Process question and provide a response
    if st.button("Ask"):
        if user_input.strip():
            response = model.predict([user_input])[0]
            confidence = model.predict_proba([user_input])[0].max() * 100

            # Fetch relevant image and video if available
            matched_row = df[df["question"].str.contains(user_input, case=False, na=False)]
            image_url = matched_row["image"].values[0] if not matched_row.empty and "image" in matched_row else None
            video_url = matched_row["video"].values[0] if not matched_row.empty and "video" in matched_row else None

            # Add to conversation history
            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("AI Tutor", f"{response} (Confidence: {confidence:.2f}%)"))

            # Display response
            st.success(f"AI Tutor: {response}")
            if image_url:
                st.image(image_url, caption="Relevant Image")
            if video_url:
                st.video(video_url)
        else:
            st.warning("Please enter a question!")

    # Display conversation history
    st.write("### Conversation History")
    for speaker, message in st.session_state.history:
        st.markdown(f"**{speaker}:** {message}")

    # Section to teach the AI Tutor
    st.write("### Teach the AI Tutor")
    with st.form("teach_form"):
        new_question = st.text_input("New Question")
        new_answer = st.text_input("New Answer")
        new_image = st.text_input("Image URL (optional)")
        new_video = st.text_input("Video URL (optional)")
        if st.form_submit_button("Teach"):
            if new_question.strip() and new_answer.strip():
                add_to_dataset(new_question, new_answer, new_image, new_video)
                st.success("Thank you! I've learned something new.")
            else:
                st.warning("Please provide both a question and an answer.")

    # About section
    with st.expander("About this app"):
        st.write("""
        - This AI Tutor can now respond with images and videos!
        - Teach the tutor new Q&A with multimedia.
        """)

# Run the app
if __name__ == "__main__":
    main()
