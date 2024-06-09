"""
Allerg-Eat: A program that takes in a user's food allergies and cravings and gives a list of ranked recipes based on these 
inputs. The heirarchy is as follows: if a recipe has the person's allergies, it is put at the bottom of the list and it will
remain lower than all of the recipes that do not have the allergens; the more allergens it has, the lower it is. Similarly,
the more cravings or words similar to the person's cravings, the higher the recipe is.

"""


# imports

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import unidecode
import ast
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# data formatting

dataframe = pd.read_json('recipe_data.json')
dataframe = dataframe.transpose()
dataframe = dataframe.reset_index()
dataframe.drop(columns=dataframe.columns[0], axis=1,  inplace=True)
df = dataframe
df = pd.DataFrame(df)

# clean out ingredients list

def ingredient_parser(ingredients):
    if isinstance(ingredients, float) and pd.isna(ingredients):
        return []
    
    measures = ['teaspoon', 't', 'tsp.', 'tablespoon', 'T', 'tbsp.', 'cup', 'c', 'ounce', 'oz', 'pint', 'quart', 'gallon', 'ml', 'liter', 'gram', 'kg', 'pound', 'lb', 'slice', 'piece', 'clove', 'stick', 'can', 'package', 'pkg', 'dash', 'pinch']
    lemmatizer = WordNetLemmatizer()
    ingred_list = []
    for i in ingredients:
        if isinstance(i, str):
            items = re.split(r'\s+|-|,.!?', i) # split by punctuation
            items = [word for word in items if word.isalpha()] # check if all alphabet
            items = [word.lower() for word in items] # make lowercase
            items = [unidecode.unidecode(word) for word in items] #remove accents
            items = [lemmatizer.lemmatize(word) for word in items] # Lemmatize words so we can compare words to measuring words
            stop_words = set(stopwords.words('english'))
            items = [word for word in items if word not in stop_words] # remove stop words
            items = [word for word in items if word not in measures] # remove measurement words
            # if items:
            #     ingred_list.append(' '.join(items))
            if items:
                ingred_list.extend(items)
        
    return ingred_list


cleaned_ingred_list = []
for lst in df['ingredients']:
    cleaned_ingred = ingredient_parser(lst)
    cleaned_ingred_list.append(cleaned_ingred)

df['cleaned_ingredients'] = cleaned_ingred_list

# use Word2Vec (CBOW) to map out relationships between words --> make your own neural network
# model = Word2Vec(sentences=cleaned_ingred_list, vector_size=100, window=5, min_count=1, workers=4)

# Create the corpus
corpus = [' '.join(ingredients) for ingredients in cleaned_ingred_list]

def generate_cbows(corpus, window_size):
    cbows = []
    for sentence in corpus:
        words = word_tokenize(sentence.lower())
        for i, target_word in enumerate(words):
            context_words = words[max(0, i - window_size):i] + words[i + 1:i + window_size + 1]
            if len(context_words) == window_size * 2:
                cbows.append((context_words, target_word))
    return cbows

window_size = 2
cbows = generate_cbows(corpus, window_size)

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear1 = nn.Linear(embed_size, vocab_size)
        
    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        return out

# Prepare the data for PyTorch
unique_words = list(set([word for sentence in cleaned_ingred_list for word in sentence]))
word_to_ix = {word: i for i, word in enumerate(unique_words)}
ix_to_word = {i: word for word, i in word_to_ix.items()}

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq if w in to_ix]  # Only retrieve index if word exists in the vocabulary
    return torch.tensor(idxs, dtype=torch.long)

# Create the model
VOCAB_SIZE = len(unique_words)
EMBED_SIZE = 10  # You can adjust this size
model = CBOWModel(VOCAB_SIZE, EMBED_SIZE)

# Loss and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the model
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for context, target in cbows:
        if target in word_to_ix:
            context_idxs = prepare_sequence(context, word_to_ix)
            target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, target_idx)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# calculate average of vector representations to get a singular value for each recipe
def calculate_average_vector(cleaned_ingred_list, model):
    avg_vectors = []
    for ingredients in cleaned_ingred_list:
        vectors = []
        for ingredient in ingredients:
            if ingredient in model.wv:
                vectors.append(model.wv[ingredient])
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            avg_vectors.append(avg_vector)
        else:
            avg_vectors.append(np.zeros(model.vector_size))
    return np.array(avg_vectors)

# use cosine similarity to figure out how closely related each recipe is to another
def calculate_cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

# rank recipes
def rank_recipes(allergies, cravings, cleaned_ingred_list, model, df):
    ranked_recipes = []
    # reference_vector = np.zeros(model.vector_size)
    reference_vector = np.zeros(EMBED_SIZE)

    # Adjust reference vector based on user's cravings
    for word in cravings:
        if word in model.wv:
            reference_vector += model.wv[word]

    # Calculate average vectors for each recipe
    average_vectors = calculate_average_vector(cleaned_ingred_list, model)

    # Calculate cosine similarity for each recipe
    for i, avg_vector in enumerate(average_vectors):
        similarity = calculate_cosine_similarity(reference_vector, avg_vector)
        ranked_recipes.append((df['title'].iloc[i], similarity))

    # Sort recipes based on cosine similarity
    ranked_recipes.sort(key=lambda x: x[1], reverse=True)

    # Adjust ranking based on allergies
    for i, (recipe_title, similarity) in enumerate(ranked_recipes):
        allergies_count = sum(ingredient in allergies for ingredient in cleaned_ingred_list[i])
        similarity_percent = round((similarity - allergies_count) * 100, 2)
        ranked_recipes[i] = (recipe_title, similarity_percent)
    
    # Return ranked recipes
    return ranked_recipes

user_allergies = ['peanuts', 'shellfish']
user_cravings = ['chocolate', 'cake']
ranked_recipes = rank_recipes(user_allergies, user_cravings, cleaned_ingred_list, model, df)
top_ten_recipes = ranked_recipes[:10]
print(top_ten_recipes)
