# %% [markdown]
# # Sentiment Analysis on IMDB Reviews using LSTM and Keras
# created by Hans Michael
# <hr>
# 
# ### Steps
# <ol type="1">
#     <li>Load the dataset (50K IMDB Movie Review)</li>
#     <li>Clean Dataset</li>
#     <li>Encode Sentiments</li>
#     <li>Split Dataset</li>
#     <li>Tokenize and Pad/Truncate Reviews</li>
#     <li>Build Architecture/Model</li>
#     <li>Train and Test</li>
# </ol>
# 
# <hr>
# <i>Import all the libraries needed</i>

# %%
import pandas as pd    # to load dataset
import numpy as np     # for mathematic equation
from nltk.corpus import stopwords   # to get collection of stopwords
from sklearn.model_selection import train_test_split       # for splitting dataset
from tensorflow.keras.preprocessing.text import Tokenizer  # to encode text to int
from tensorflow.keras.preprocessing.sequence import pad_sequences   # to do padding or truncating
from tensorflow.keras.models import Sequential     # the model
from tensorflow.keras.layers import Embedding, LSTM, Dense # layers of the architecture
from tensorflow.keras.callbacks import ModelCheckpoint   # save model
from tensorflow.keras.models import load_model   # load saved model
import re

# %% [markdown]
# <hr>
# <i>Preview dataset</i>

# %%
data = pd.read_csv('IMDB Dataset.csv')

print(data)

# %% [markdown]
# <hr>
# <b>Stop Word</b> is a commonly used words in a sentence, usually a search engine is programmed to ignore this words (i.e. "the", "a", "an", "of", etc.)
# 
# <i>Declaring the english stop words</i>

# %%
english_stops = set(stopwords.words('english'))

# %% [markdown]
# <hr>
# 
# ### Load and Clean Dataset
# 
# In the original dataset, the reviews are still dirty. There are still html tags, numbers, uppercase, and punctuations. This will not be good for training, so in <b>load_dataset()</b> function, beside loading the dataset using <b>pandas</b>, I also pre-process the reviews by removing html tags, non alphabet (punctuations and numbers), stop words, and lower case all of the reviews.
# 
# ### Encode Sentiments
# In the same function, I also encode the sentiments into integers (0 and 1). Where 0 is for negative sentiments and 1 is for positive sentiments.

# %%
def load_dataset():
    df = pd.read_csv('IMDB Dataset.csv')
    x_data = df['review']       # Reviews/Input
    y_data = df['sentiment']    # Sentiment/Output

    # PRE-PROCESS REVIEW
    x_data = x_data.replace({'<.*?>': ''}, regex = True)          # remove html tag
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)     # remove non alphabet
    x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
    x_data = x_data.apply(lambda review: [w.lower() for w in review])   # lower case
    
    # ENCODE SENTIMENT -> 0 & 1
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data

x_data, y_data = load_dataset()

print('Reviews')
print(x_data, '\n')
print('Sentiment')
print(y_data)

# %% [markdown]
# <hr>
# 
# ### Split Dataset
# In this work, I decided to split the data into 80% of Training and 20% of Testing set using <b>train_test_split</b> method from Scikit-Learn. By using this method, it automatically shuffles the dataset. We need to shuffle the data because in the original dataset, the reviews and sentiments are in order, where they list positive reviews first and then negative reviews. By shuffling the data, it will be distributed equally in the model, so it will be more accurate for predictions.

# %%
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)

print('Train Set')
print(x_train, '\n')
print(x_test, '\n')
print('Test Set')
print(y_train, '\n')
print(y_test)

# %% [markdown]
# <hr>
# <i>Function for getting the maximum review length, by calculating the mean of all the reviews length (using <b>numpy.mean</b>)</i>

# %%
def get_max_length():
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))

# %% [markdown]
# <hr>
# 
# ### Tokenize and Pad/Truncate Reviews
# A Neural Network only accepts numeric data, so we need to encode the reviews. I use <b>tensorflow.keras.preprocessing.text.Tokenizer</b> to encode the reviews into integers, where each unique word is automatically indexed (using <b>fit_on_texts</b> method) based on <b>x_train</b>. <br>
# <b>x_train</b> and <b>x_test</b> is converted into integers using <b>texts_to_sequences</b> method.
# 
# Each reviews has a different length, so we need to add padding (by adding 0) or truncating the words to the same length (in this case, it is the mean of all reviews length) using <b>tensorflow.keras.preprocessing.sequence.pad_sequences</b>.
# 
# 
# <b>post</b>, pad or truncate the words in the back of a sentence<br>
# <b>pre</b>, pad or truncate the words in front of a sentence

# %%
# ENCODE REVIEW
token = Tokenizer(lower=False)    # no need lower, because already lowered the data in load_data()
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)

max_length = get_max_length()

x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

total_words = len(token.word_index) + 1   # add 1 because of 0 padding

print('Encoded X Train\n', x_train, '\n')
print('Encoded X Test\n', x_test, '\n')
print('Maximum review length: ', max_length)

# %% [markdown]
# <hr>
# 
# ### Build Architecture/Model
# <b>Embedding Layer</b>: in simple terms, it creates word vectors of each word in the <i>word_index</i> and group words that are related or have similar meaning by analyzing other words around them.
# 
# <b>LSTM Layer</b>: to make a decision to keep or throw away data by considering the current input, previous output, and previous memory. There are some important components in LSTM.
# <ul>
#     <li><b>Forget Gate</b>, decides information is to be kept or thrown away</li>
#     <li><b>Input Gate</b>, updates cell state by passing previous output and current input into sigmoid activation function</li>
#     <li><b>Cell State</b>, calculate new cell state, it is multiplied by forget vector (drop value if multiplied by a near 0), add it with the output from input gate to update the cell state value.</li>
#     <li><b>Ouput Gate</b>, decides the next hidden state and used for predictions</li>
# </ul>
# 
# <b>Dense Layer</b>: compute the input with the weight matrix and bias (optional), and using an activation function. I use <b>Sigmoid</b> activation function for this work because the output is only 0 or 1.
# 
# The optimizer is <b>Adam</b> and the loss function is <b>Binary Crossentropy</b> because again the output is only 0 and 1, which is a binary number.

# %%
# ARCHITECTURE
EMBED_DIM = 32
LSTM_OUT = 64

model = Sequential()
model.add(Embedding(total_words, EMBED_DIM, input_length = max_length))
model.add(LSTM(LSTM_OUT))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print(model.summary())

# %% [markdown]
# <hr>
# 
# ### Training
# For training, it is simple. We only need to fit our <b>x_train</b> (input) and <b>y_train</b> (output/label) data. For this training, I use a mini-batch learning method with a <b>batch_size</b> of <i>128</i> and <i>5</i> <b>epochs</b>.
# 
# Also, I added a callback called **checkpoint** to save the model locally for every epoch if its accuracy improved from the previous epoch.

# %%
checkpoint = ModelCheckpoint(
    'models/LSTM.h5',
    monitor='accuracy',
    save_best_only=True,
    verbose=1
)

# %%
model.fit(x_train, y_train, batch_size = 128, epochs = 5, callbacks=[checkpoint])

# %% [markdown]
# <hr>
# 
# ### Testing
# To evaluate the model, we need to predict the sentiment using our <b>x_test</b> data and comparing the predictions with <b>y_test</b> (expected output) data. Then, we calculate the accuracy of the model by dividing numbers of correct prediction with the total data. Resulted an accuracy of <b>86.63%</b>

# %%
y_pred = model.predict_classes(x_test, batch_size = 128)

true = 0
for i, y in enumerate(y_test):
    if y == y_pred[i]:
        true += 1

print('Correct Prediction: {}'.format(true))
print('Wrong Prediction: {}'.format(len(y_pred) - true))
print('Accuracy: {}'.format(true/len(y_pred)*100))

# %% [markdown]
# ---
# 
# ### Load Saved Model
# 
# Load saved model and use it to predict a movie review statement's sentiment (positive or negative).

# %%
loaded_model = load_model('models/LSTM.h5')

# %% [markdown]
# Receives a review as an input to be predicted

# %%
review = str(input('Movie Review: '))

# %% [markdown]
# The input must be pre processed before it is passed to the model to be predicted

# %%
# Pre-process input
regex = re.compile(r'[^a-zA-Z\s]')
review = regex.sub('', review)
print('Cleaned: ', review)

words = review.split(' ')
filtered = [w for w in words if w not in english_stops]
filtered = ' '.join(filtered)
filtered = [filtered.lower()]

print('Filtered: ', filtered)

# %% [markdown]
# Once again, we need to tokenize and encode the words. I use the tokenizer which was previously declared because we want to encode the words based on words that are known by the model.

# %%
tokenize_words = token.texts_to_sequences(filtered)
tokenize_words = pad_sequences(tokenize_words, maxlen=max_length, padding='post', truncating='post')
print(tokenize_words)

# %% [markdown]
# This is the result of the prediction which shows the **confidence score** of the review statement.

# %%
result = loaded_model.predict(tokenize_words)
print(result)

# %% [markdown]
# If the confidence score is close to 0, then the statement is **negative**. On the other hand, if the confidence score is close to 1, then the statement is **positive**. I use a threshold of **0.7** to determine which confidence score is positive and negative, so if it is equal or greater than 0.7, it is **positive** and if it is less than 0.7, it is **negative**

# %%
if result >= 0.7:
    print('positive')
else:
    print('negative')

# %%



