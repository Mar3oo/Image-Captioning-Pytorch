#This code now integrates the Flickr8k dataset for both training and testing,--
#--handling caption loading, image processing, and batch sampling appropriately.

import csv
import random
import nltk
import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
#from vocabulary import vocabulary 
from vocabulary_mine import Vocab

#from tqdm import tqdm
#import random
#import json



def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               flickr8k_loc='./flickr8k'):
    """Returns the data loader for the Flickr8k dataset.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      flickr8k_loc: The location of the folder containing the Flickr8k dataset.
    """
    
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."
    if vocab_from_file==False: assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file=>Captions.
    if mode == 'train':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(flickr8k_loc, 'Images/')
        annotations_file = os.path.join(flickr8k_loc, 'captions.txt')
    if mode == 'test':
        assert batch_size==1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(flickr8k_loc, 'Images/')
        annotations_file = os.path.join(flickr8k_loc, 'captions.txt')

    # Flickr8k caption dataset.
    dataset = FlickrDataset(transform=transform,
                            mode=mode,
                            batch_size=batch_size,
                            vocab_threshold=vocab_threshold,
                            vocab_file=vocab_file,
                            start_word=start_word,
                            end_word=end_word,
                            unk_word=unk_word,
                            annotations_file=annotations_file,
                            vocab_from_file=vocab_from_file,
                            img_folder=img_folder)

    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        
        #debugging code to check if batch size is less than dataset size. 
        assert dataset.batch_size <= len(dataset), "Batch size exceeds dataset size."

        # data loader for Flickr8k dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=True))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
        
        
    #print(f"Mode: {mode}")
    #print(f"Number of samples in dataset: {len(dataset)}")
    #print(f"Batch size: {dataset.batch_size}")
    #print(f"Sampled indices: {indices}")
    


    return data_loader



class FlickrDataset(data.Dataset):
    
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocab(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.load_captions(annotations_file)
        else:
            self.load_test_images(annotations_file)
        
    def load_captions(self, annotations_file): # annotation_file => captions.txt
        """Load and process the captions from the captions.txt file."""
        self.captions = {}
        self.image_ids = []
        with open(annotations_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header line ('image', 'caption')
            for line in f:
                img_id, caption = line.strip().split(',', 1) # Split the line at the first comma
                if img_id not in self.captions:
                    self.captions[img_id] = []
                    self.image_ids.append(img_id)
                self.captions[img_id].append(caption) # handles multiple captions per image id 
                
        #print(f"Number of images: {len(self.image_ids)}")
        #print(f"Number of captions: {len(self.captions)}")

        #print(self.image_ids[100])
        #print(self.captions[self.image_ids[100]])
        print(f'Number of images: {len(self.image_ids)}')
        print(f'Number of captions: {len(self.captions)}')

        print('Obtaining caption lengths...')
        all_tokens = [nltk.tokenize.word_tokenize(str(caption).lower()) for caption in self.captions.values()]
        self.caption_lengths = [len(token) for token in all_tokens]
        print(f'Obtaining caption lengths...Done and caption lengths: {len(self.caption_lengths)}')
        
        caption_lengths = [len(caption) for caption in self.captions.values()]
        self.max_len = max(caption_lengths)
        
    def load_test_images(self, annotations_file): # annotation_file => captions.txt
        """Load the image paths for testing."""
        self.paths = []
        with open(annotations_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header line ('image', 'caption')
            for line in f:
                img_id, _ = line.strip().split(',', 1)  # Split the line at the first comma
                self.paths.append(img_id)
                
        print(self.paths[115])
        print(f"Number of images: {len(self.paths)}")

    def __getitem__(self, index):
        if self.mode == 'train':
            
            
            img_id = self.image_ids[index]
            caption = self.captions[img_id]
            path = img_id


            # Validate img_id
            #if img_id not in self.captions:
            #    raise ValueError(f"Image ID {img_id} not found in captions!")
            #captions = self.captions[img_id]
            
            # Validate image path
            #path = img_id
            #img_path = os.path.join(self.img_folder, path)
            #if not os.path.exists(img_path):
            #    raise FileNotFoundError(f"Image file {img_path} does not exist!")


            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            caption_tokenized = []
            caption_tokenized.append(self.vocab(self.vocab.start_word))
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption_tokenized.extend([self.vocab(token) for token in tokens])
            caption_tokenized.append(self.vocab(self.vocab.end_word))
            caption_tokenized += [2] * (self.max_len - len(caption_tokenized))
            caption_tokenized_padded = torch.Tensor(caption_tokenized).long()
            

            return image, caption_tokenized_padded

        else:
            path = self.paths[index] 

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            return orig_image, image

    def get_train_indices(self):
        
        # Get a list of unique caption lengths and their counts
        unique_lengths, counts = np.unique(self.caption_lengths, return_counts=True)

        # Prioritize sampling lengths with higher counts
        probabilities = counts / np.sum(counts) 
        
        # Select a caption length based on the probabilities
        sel_length = np.random.choice(unique_lengths, p=probabilities)
        
        # Find indices where caption_lengths match the selected length.
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        
        # Check if there are enough indices for the batch size
        if len(all_indices) < self.batch_size:
            # try again 
            self.get_train_indices() 
            
            #print(f"Warning: Not enough indices for the selected caption length {sel_length}. Only {len(all_indices)} available.")
            # If there are not enough indices, just sample all available ones
            indices = list(np.random.choice(all_indices, size=len(all_indices), replace=False))
        else:
            # Otherwise, sample exactly the batch_size number of indices
            indices = list(np.random.choice(all_indices, size=self.batch_size, replace=False))
        
        # Check if the indices are within the valid range
        assert all(0 <= i < len(self.caption_lengths) for i in indices), "Some indices are out of range."
        
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.image_ids)
        else:
            return len(self.paths)

