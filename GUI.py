from data_loader import get_loader
from torchvision import transforms
import torch
import os
import torch
from model import EncoderCNN, DecoderRNN
import streamlit as st
from PIL import Image
from gtts import gTTS
import base64


# Define a transform to pre-process the testing images.
transform_test = transforms.Compose([ 
    transforms.Resize((256,256)),                          # smaller edge of image resized to 256
    transforms.RandomCrop((224,224)),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Create the data loader.
data_loader = get_loader(transform=transform_test,    
                         mode='test') 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Specify the saved models to load.
encoder_file ='encoder-2.pkl' 
decoder_file = 'decoder-2.pkl'

# Select appropriate values for the Python variables below.
embed_size = 256
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)


#4: Complete the function.
def clean_sentence(output):
    sentence = ""
    for i in output:
        word = data_loader.dataset.vocab.idx2word[i]
        if (word == data_loader.dataset.vocab.start_word):
            continue
        elif (word == data_loader.dataset.vocab.unk_word):
            continue
        elif (word == data_loader.dataset.vocab.end_word):
            break
        else:
            
            sentence = sentence + " " + word
    return sentence

# Streamlit UI
st.title("Image Captioning and Audio Generator")
st.write("Upload an image to generate a caption and its corresponding audio.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Transform the image
    image_tensor = transform_test(image).unsqueeze(0).to(device)

    # Get features from encoder
    features = encoder(image_tensor).unsqueeze(1)

    # Generate caption
    output = decoder.sample(features)

    # Clean the sentence
    sentence = clean_sentence(output)

    st.subheader("Generated Caption:")
    st.write(sentence[4:-17])

    # Generate audio
    tts = gTTS(text=sentence[4:-17], lang='en')
    audio_path = "generated_audio.mp3"
    tts.save(audio_path)

    # Provide audio for download
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", start_time=0)

    # Provide a download link
    b64 = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="caption_audio.mp3">Download Audio</a>'
    st.markdown(href, unsafe_allow_html=True)
