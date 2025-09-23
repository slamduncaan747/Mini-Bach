import torch
import numpy as np
import pickle
from music21 import *
from utility import visualize_as_matrix, play_dataset_element, voice_ranges
from model import FeedForwardNN
import os

def probabilities_to_one_hot(model_output):
    batch_size = model_output.shape[0]
    
    reshaped_output = model_output.view(batch_size, 70, 64)
    
    alto_probs = reshaped_output[:, 0:21, :]   
    tenor_probs = reshaped_output[:, 21:42, :] 
    bass_probs = reshaped_output[:, 42:70, :]   
    
    alto_indices = torch.argmax(alto_probs, dim=1)
    tenor_indices = torch.argmax(tenor_probs, dim=1)
    bass_indices = torch.argmax(bass_probs, dim=1)
    
    final_output = torch.zeros_like(reshaped_output)
    
    final_output.scatter_(1, (alto_indices).unsqueeze(1), 1.0)
    final_output.scatter_(1, (tenor_indices + 21).unsqueeze(1), 1.0)
    final_output.scatter_(1, (bass_indices + 42).unsqueeze(1), 1.0)
            
    return final_output.view(batch_size, -1)

# Get model
model = FeedForwardNN(input_size=1344, hidden_size=200, output_size=4480, dropout_rate=0)
model.load_state_dict(torch.load('checkpoints/checkpoint1000.pth'))
model.eval()

# Choose a random dataset element, and generate ouput
with open('chorale_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
random_index = np.random.randint(0, len(dataset))
input_tensor = torch.tensor(dataset[random_index]['input'], dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    output_tensor = model(input_tensor) 

# Convert from logits with probability distribution to one-hot piano roll
output_tensor = probabilities_to_one_hot(output_tensor)
output_vector = output_tensor.squeeze(0).numpy()

# Make dataset element
dataset_element = {
    'chorale': dataset[random_index]['chorale'],
    'measures': dataset[random_index]['measures'],
    'input': dataset[random_index]['input'],
    'output': output_vector
}

# Visualize piano roll, convert to midi, and play midi audio
visualize_as_matrix(dataset_element, voice_ranges)
midi = play_dataset_element(dataset_element, voice_ranges, bpm=100)
os.makedirs('samples', exist_ok=True)
midi.write('midi', fp='samples/chorale.mid')
