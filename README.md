# Spam Detection using Universal Sentence Encoder and Neural Networks

Overview
This project demonstrates a spam detection system using the Universal Sentence Encoder (USE) and a simple neural network implemented with TensorFlow. The Universal Sentence Encoder is used to convert textual data into fixed-size embeddings, and a neural network is trained to classify whether a given text is spam or not.

##Table of Contents
Introduction
Project Structure
Dependencies
Installation
Usage
Model Architecture
Training
Evaluation
Inference
Custom Input Prediction
License

##Introduction
Spam detection is a common problem in natural language processing, and this project addresses it by utilizing the Universal Sentence Encoder, a pre-trained model developed by Google, to convert text data into embeddings. These embeddings are then used as input to a neural network that is trained to classify messages as either spam or not spam.

##Project Structure
main.py: The main script that loads the USE, constructs the neural network, and trains the model.
utils.py: Contains utility functions for data loading, preprocessing, and evaluation.
trained_model/: A directory to store the trained model.
data/: Placeholder for your dataset (replace with your actual dataset).

##Dependencies
TensorFlow
TensorFlow Hub
scikit-learn
Install dependencies using:
pip install tensorflow tensorflow-hub scikit-learn

##Installation
Clone the repository:
git clone https://github.com/vikishan13/Email-spam-detection-using-Transfer-Learning.git

Navigate to the project directory:
cd spam-detection

##Install dependencies:
pip install -r requirements.txt
Usage
Prepare your dataset in the data/ directory.
Run the main script:

python main.py
Model Architecture
The neural network architecture consists of a single dense layer with ReLU activation followed by a dropout layer for regularization and a final dense layer with sigmoid activation for binary classification.

plaintext
Copy code
Input (USE Embeddings) -> Dense(64, ReLU) -> Dropout -> Dense(1, Sigmoid)
Training
The model is trained using the training set, and the training process involves minimizing the binary crossentropy loss using the Adam optimizer.


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, labels_train, epochs=5, batch_size=32, validation_split=0.2)
Evaluation
The model is evaluated on the test set, and accuracy along with a classification report is printed.


accuracy = accuracy_score(labels_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(labels_test, predictions))
Inference
To use the trained model for inference on new data, load the model and make predictions:


model = tf.keras.models.load_model('trained_model')
predictions = model.predict(X_new_data)
Custom Input Prediction
You can predict whether a custom input is spam or not using the provided script:


python predict_custom_input.py
License
This project is licensed under the MIT License.

