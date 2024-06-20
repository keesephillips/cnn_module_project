import streamlit as st
from PIL import Image
import numpy as np
import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Get class names associated with labels
class_names = ['Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier', 'Appenzeller', 'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull', 'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever', 'Chihuahua', 'Dandie_Dinmont', 'Doberman', 'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher', 'Eskimo_dog', 'French_bulldog', 'German_shepherd', 'German_short', 'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog', 'Mexican_hairless', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed', 'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog', 'Shih', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel', 'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'Yorkshire_terrier', 'affenpinscher', 'basenji', 'basset', 'beagle', 'black', 'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly', 'dhole', 'dingo', 'flat', 'giant_schnauzer', 'golden_retriever', 'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'otterhound', 'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier', 'soft', 'standard_poodle', 'standard_schnauzer', 'toy_poodle', 'toy_terrier', 'vizsla', 'whippet', 'wire']

def load_image(image_file):
    img = Image.open(image_file)
    return img

if __name__ == '__main__':
    st.header('Dog Breed Classifier', divider='red')
    
    col1, col2 = st.columns(2)

    puppy = Image.open('assets/puppy.jpg')
    col1.image(puppy, use_column_width=True)

    dog_fetch = Image.open('assets/dog_fetch.jpg')
    col2.image(dog_fetch, use_column_width=True)
    
    uploaded_image = st.file_uploader("Choose an image file", type=["jpg", "png"])
    
    st.divider()
    
    # Using "with" notation
    with st.sidebar:
        model_type = st.radio(
            "Choose the dog breed classifier type",
            ("Primary Breed", "Breed Percentages")
        )
        threshold = st.slider("Minimum Value", 0, 100, 5)

    if uploaded_image is not None:
        img = load_image(uploaded_image)
        img.save('assets/user/image.jpg')
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model = torch.load('models/resnet50.pt',map_location=device)

        data_dir = 'assets/user'

        # Set up transformations for training and validation (test) data
        # For training data we will do randomized cropping to get to 224 * 224, randomized horizontal flipping, and normalization
        # For test set we will do only center cropping to get to 224 * 224 and normalization
        data_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Create Datasets for training and validation sets
        image = data_transforms(img)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.eval()  # Set to evaluation mode

        with torch.no_grad():
            output = model(image.unsqueeze(0))  # Unsqueeze to add batch dimension
            predicted_class = torch.argmax(output).item()
            predicted_class = class_names[predicted_class].replace('_', ' ').title()

            st.subheader(f'Predicted Breed:\n{predicted_class}')
            st.image(img, caption="Uploaded image of dog", use_column_width=True)
            
            if model_type == 'Breed Percentages':
                probs = F.softmax(output,dim=1) # get the predictions using softmax
                probs = pd.DataFrame(probs.numpy()[0]).sort_values(by=0, ascending=False) # sort predictions in descending order
                probs = probs[probs[0] >= (threshold / 100)]
                

                for index, row in probs.iterrows():
                    dog_breeds = st.columns(2)
                    dog_breeds[0].write(class_names[index].replace('_', ' ').title())
                    dog_breeds[1].write('{:.2f}%'.format(row[0]*100))

