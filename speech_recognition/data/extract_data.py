import os
from os.path import isdir
from collections import defaultdict
import numpy as np
import librosa


#converts raw audio to padded mfcc
def audio_to_mfcc(audio_path):
    freq, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(freq, sr=16000)
    pad_width = 32 - mfcc.shape[1]
    padded_mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return padded_mfcc

#getting list of words
words = [word for word in os.listdir('../../data') if isdir('../../data/{}'.format(word))]
#creating directories in data for training, validation, and test sets
for fold in ['train', 'validation', 'test']:
    os.mkdir('../../data/{}'.format(fold))
    #creating directories in each for each word (background noise only in train)
    for word in words:
        if  word != '_background_noise_':
            os.mkdir('../../data/{}/{}'.format(fold, word))
            os.mkdir('../../data/{}/{}/raw_audio'.format(fold, word))


#reading validation and testing lists and storing in separate dictionaries
#key: word
#value: list of audio files for that word
validation_dict = defaultdict(list)
testing_dict = defaultdict(list)

with open('../../data/validation_list.txt') as f:
    for line in f:
        word, filename = line.rstrip('\n').split('/')
        validation_dict[word].append(filename)

with open('../../data/testing_list.txt') as f:
    for line in f:
        word, filename = line.rstrip('\n').split('/')
        testing_dict[word].append(filename)


#moving audio files to their respective directories and extracting mfcc
#tqdm shows progress bar
for i, word in enumerate(words):
    if word == '_background_noise_':
        continue
    print('Processing word {} of {}...'.format(i+1, len(words)-1))
    
    mfcc_dict = {
            'train': [],
            'validation': [],
            'test': []
            }
    
    for file in os.listdir('../../data/{}'.format(word)):
        #checks if the files are in validation or testing lists
        if file in validation_dict[word]:
            fold = 'validation'
        elif file in testing_dict[word]:
            fold = 'test'
        else:
            fold = 'train'
        
        #extracting data
        initial_audio_path = '../../data/{}/{}'.format(word, file)
        target_audio_path = '../../data/{}/{}/raw_audio/{}'.format(fold, word, file)
        mfcc = audio_to_mfcc(initial_audio_path)
        mfcc_dict[fold].append(mfcc)
        #moving raw audio
        os.rename(initial_audio_path, target_audio_path)
        
    #saving extracted information   
    for fold in mfcc_dict.keys():
        target_mfcc_path = '../../data/{}/{}/mfcc.npy'.format(fold, word)
        np.save(target_mfcc_path, mfcc_dict[fold])
    #deleting original empty directories    
    os.rmdir('../../data/{}'.format(word))

print('Done!')