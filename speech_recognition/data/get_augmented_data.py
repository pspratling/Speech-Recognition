import os
import numpy as np
import librosa

#all clips are sampled with samplerate=16000
SAMPLERATE = 16000
#words to add augmented data to
WORDS = ['no', 'go', 'down']
#background noise files to use
SILENCE_FILENAMES = [file for file in os.listdir('../../data/_background_noise_') if file[-4:]=='.wav']
#number of augmented clips for each word for each background clip
NUM_SAMPLES = 250


def create_augmented_sample(word_path, silence_path, offset, scalar):
    """
    Creates new word sample from an existing sample and a clip of silence
    
    word_path: filepath to word sample
    silence_path: filepath to background noise file
    offset: offset used to extract silence
    scalar: multiple to determine how loud background noise is
    
    returns mfcc
    """
    freq_word, _ = librosa.load(word_path, sr=None)
    freq_silence, _ = librosa.load(silence_path, offset=offset, duration=1, sr=None)
    if len(freq_word) != 16000:
        pad_length = 16000 - len(freq_word)
        freq_word = np.append(freq_word, np.zeros(pad_length))
    freq_total = freq_word + scalar*freq_silence
    mfcc = librosa.feature.mfcc(freq_total, sr=SAMPLERATE)
    return mfcc


def get_augmented_clips(word, silence_filename, num_samples):
    """
    Creates augmented clips for a given background noise file and word
    
    returns array of mfcc's
    """
    silence_path = '../../data/_background_noise_/{}'.format(silence_filename)
    duration = librosa.get_duration(filename=silence_path)
    offsets = np.random.uniform(0, duration-1, size=num_samples)
    if silence_filename in ['white_noise.wav', 'pink_noise.wav']:
        scalars = np.random.uniform(.05, .2, size=num_samples)
    else:
        scalars = np.ones(num_samples)
    mfcc_list = []
    for i in range(num_samples):      
        random_sample = np.random.choice(os.listdir('../../data/train/{}/raw_audio'.format(word)))
        word_path = '../../data/train/{}/raw_audio/{}'.format(word, random_sample)
        mfcc = create_augmented_sample(word_path, silence_path, offsets[i], scalars[i])
        mfcc_list.append(mfcc)
    return np.array(mfcc_list)


#creating and saving augmented samples
for i, word in enumerate(WORDS):
    print('Augmenting samples from word {} of {}...'.format(i+1, len(WORDS)))
    mfcc_list = np.load('../../data/train/{}/mfcc.npy'.format(word))
    for filename in SILENCE_FILENAMES:
        mfcc_augmented = get_augmented_clips(word, filename, num_samples=NUM_SAMPLES)
        mfcc_list = np.vstack([mfcc_list, mfcc_augmented])
    np.save('../../data/train/{}/mfcc_augmented.npy'.format(word), mfcc_list)
print('Done!')