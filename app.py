from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed, Dropout

# Initialize Flask app
app = Flask(__name__)

# Define the LSTM model
def define_model(vocab_size, max_length):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(4096,)))  # Change to 4096
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Load the trained LSTM model
model_path = 'caption_model.keras'  # Update with the correct path if needed
vocab_size = 10000  # Update this with your actual vocab size
max_length = 34  # Replace with the actual max_length used in your model

if os.path.exists(model_path):
    lstm_model = tf.keras.models.load_model(model_path)
else:
    lstm_model = define_model(vocab_size, max_length)

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocess the image before feeding it to the CNN model
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape[-1] == 4:
        img = img[..., :3]  # Remove alpha channel if exists
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Use VGG16 to extract features from the fc1 layer (1920-dimensional vector)
def extract_features(image):
    vgg_model = tf.keras.applications.VGG16(include_top=True, weights='imagenet')
    model = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc1').output)  # Use 'fc1' layer for 1920-dimensional output
    features = model.predict(image)
    return features

# Function to generate a caption using the LSTM model
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Define route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define route to upload and caption image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded image
    image_path = os.path.join('static/uploads', file.filename)
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    file.save(image_path)

    # Preprocess the image
    img = preprocess_image(image_path)

    # Extract features using VGG16
    image_feature = extract_features(img)

    # Generate caption
    caption = generate_caption(lstm_model, tokenizer, image_feature, max_length)

    return render_template('result.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
