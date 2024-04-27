import pickle
import numpy as np
import re
from flask import Flask, request, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()


app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('index.html')


with open('leme.pickle', 'rb') as handle:
    le = pickle.load(handle)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

loaded_model = load_model('./saved_weights/model.h5')


def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]
    
    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    
    text = text.split()

    text=[y.lower() for y in text]
    
    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )
    
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    try:
        sentence= lower_case(sentence)
        sentence= remove_stop_words(sentence)
        sentence= Removing_numbers(sentence)
        sentence= Removing_punctuations(sentence)
        sentence= Removing_urls(sentence)
        sentence= lemmatization(sentence)
        return sentence
    except Exception as e:
        print(f"Error in normalization: {str(e)}")
        return None


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sentence = request.form.get('sentence')
        if not sentence:
            raise ValueError('Input sentence is empty.')

        app.logger.info(f"Received sentence: {sentence}")

        normalized = normalized_sentence(sentence)
        app.logger.info(f"normalized_sentenc: {normalized}") 
        tokenized_sentence = tokenizer.texts_to_sequences([normalized])
        padded_sentence = pad_sequences(tokenized_sentence, maxlen=229, truncating='pre')
        prediction = loaded_model.predict(padded_sentence)
        predicted_class = le.inverse_transform(np.argmax(prediction, axis=-1))[0]
        probability = np.max(prediction)

        app.logger.info(f"Prediction: {predicted_class} with {probability * 100:.2f}% probability")

        return render_template('index.html', prediction_text=f'{predicted_class} with {probability * 100:.2f}% probability')

    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return render_template('index.html', prediction_text='Error predicting sentiment. Please try again.')

if __name__ == "__main__":
    app.run(debug=True)
