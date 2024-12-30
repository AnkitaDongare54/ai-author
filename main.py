import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load the dataset
def load_data():
    data_path = "data/dataset.csv"
    df = pd.read_csv(data_path)
    return df

# Train the model
def train_model(df):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df['question'], df['answer'])
    return model

# AI Tutor interaction
def ai_tutor(model):
    print("AI Tutor: Hello! Ask me a question. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("AI Tutor: Goodbye!")
            break
        response = model.predict([user_input])[0]
        print(f"AI Tutor: {response}")

# Main function
if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    ai_tutor(model)
