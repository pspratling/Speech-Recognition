import os
import numpy as np
import librosa

#all clips are sampled with samplerate=16000
SAMPLERATE = 16000

def extract_clip(filename, duration, offset):
    """
    Extracts 1 second clip from a given file
    
    filename: background noise filename to extract from
    duration: length of the background noise audio clip
    offset: where in the clip to start extracting from
    
    returns mfcc
    """
    freq, _ = librosa.load(filename, offset=offset, duration=1, sr=None)
    mfcc = librosa.feature.mfcc(freq, sr=SAMPLERATE)
    return mfcc

def get_silence_samples(filename, num_samples):
    """
    Extracts 1 second clips randomly from a given background noise file
    
    filename: background noise filename to extract from
    num_samples: desired number of 1 second clips to extract
    
    returns array of extracted samples in mfcc form
    """
    
    duration = librosa.get_duration(filename=filename)
    offsets = np.random.uniform(0, duration-1, size=num_samples)
    mfcc_list = []
    for i in range(num_samples):
        mfcc = extract_clip(filename, duration, offsets[i])
        mfcc_list.append(mfcc)
    return np.array(mfcc_list)



silence_filenames = [file for file in os.listdir('../../data/_background_noise_') if file[-4:]=='.wav']

#looping through background noise clips and saving MFCCs for use in models
for fold, num_samples in zip(['train', 'validation', 'test'], [550, 75, 75]):
    print('Extracting silence clips for {}...'.format(fold))
    mfcc_list = np.array([]).reshape(0, 20 , 32)
    for filename in silence_filenames:
        mfcc = get_silence_samples('../../data/_background_noise_/{}'.format(filename),
                                         num_samples=num_samples)
        mfcc_list = np.vstack([mfcc_list, mfcc])
        
    os.mkdir('../../data/{}/silence'.format(fold))
    target_mfcc_path = '../../data/{}/silence/mfcc.npy'.format(fold)
    np.save(target_mfcc_path, mfcc_list)
print('Done!')