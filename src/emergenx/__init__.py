
import random

import altair as alt
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from io import BytesIO
import base64
from PIL import Image



def relu(X):
    return np.maximum(X, 0)

def softmax(X):
    return np.exp(X) / np.exp(X).sum(axis=1, keepdims=True)

def get_layer_output(model, X, layer):
    output = X
    
    for i in range(layer - 1):
        z = np.dot(output, model.coefs_[i]) + model.intercepts_[i]
        
        if i < model.n_layers_ - 2:
            output = relu(z)
        else:
            output = softmax(z)
        
    return output
def embed_activations(X_train, y_train, nn, layers):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    dfs = []
    
    for i in layers:
        activations = get_layer_output(nn, X_train, i)
        embedded = tsne.fit_transform(activations)
        dfs.append(pd.DataFrame({
            'x': embedded[:,0],
            'y': embedded[:,1],
            'layer': i,
            'label': y_train
        }))
    
    return pd.concat(dfs)

def encode_images(matrix, prefix):
    paths = []
    
    for i, row in enumerate(matrix):
        d = int(np.sqrt(len(row)))
        matrix = row.reshape((d, d)).astype(np.uint8)
        
        img = Image.fromarray(matrix)
        
        with BytesIO() as buffer:
            img.save(buffer, 'png')
            data = base64.encodebytes(buffer.getvalue()).decode('utf-8')
            
        
        paths.append(f'data:image/png;base64,{data}')
        
    return paths

def embed_activations(X_train, y_train, nn, layers):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    dfs = []
    
    for i in layers:
        activations = get_layer_output(nn, X_train, i)
        embedded = tsne.fit_transform(activations)
        dfs.append(pd.DataFrame({
            'x': embedded[:,0],
            'y': embedded[:,1],
            'layer': i,
            'label': y_train,
            'image': encode_images(X_train, 'train')
        }))
    
    return pd.concat(dfs)

def ploty_plot(X_train, y_train, nn, layers,subset=2000):
    X_train = X_train[:2000]
    y_train = y_train[:2000]
    df_embedded = embed_activations(X_train, y_train, nn, layers)
    alt.data_transformers.enable('default', max_rows=None)
    ploty = alt.Chart(df_embedded).mark_circle().encode(
    x='x',
    y='y',
    color='label:N',
    column='layer:N',
    tooltip=['label', 'image']).interactive()
    return ploty