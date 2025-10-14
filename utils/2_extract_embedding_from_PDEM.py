import audeer
import audonnx
import numpy as np
import audinterface
import os
import audb
import audformat
import pandas as pd


def rename_file(old_path, new_name):
    try:
        directory, old_filename = os.path.split(old_path)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"File name successfully changed:{old_path} -> {new_path}")
    except OSError as e:
        print(f"Wrong with changing file name:{e}")


# Below is a github repo link of PDEM for your reference in which you can find the documentation for using PDEM. 
# Link：https://github.com/audeering/w2v2-how-to?tab=readme-ov-file

# This is the PDEM model url. 
url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
cache_root = audeer.mkdir('pdem_cache')
model_root = audeer.mkdir('pdem_model')

# You don't need download the PDEM model again since you have done it previous (4) script.

# archive_path = audeer.download_url(url, cache_root, verbose=True)
# audeer.extract_archive(archive_path, model_root)

# Load model from model_root using audonnx package.
model = audonnx.load(model_root)

# Set audio sampling rate. 
# The sampling rate of E-DAIC audio data is 16000 indeed, so you don't need to change this setting.
sampling_rate = 16000

# Define model interface, output is PDEM embedding. This format is fixed defined by audinterface pacakage.
# You can adjust the parameter num_workers, the bigger the faster, as long as your computer can tolerate.
hidden_states = audinterface.Feature(
    model.labels('hidden_states'),
    process_func=model,
    process_func_args={
        'outputs': 'hidden_states',
    },
    sampling_rate=sampling_rate,    
    resample=True,    
    num_workers=8,
    verbose=True,
)

# Step 2: Indicate the train, dev and test index files respectively which already got from scripts 1-3.
train_index_file = '/media/legalalien/Data1/AVEC2019_DAIC_WOZ/split_audio_labels/mapped_dur_train.csv'
dev_index_file = '/media/legalalien/Data1/AVEC2019_DAIC_WOZ/split_audio_labels/mapped_dur_dev.csv'
test_index_file = '/media/legalalien/Data1/AVEC2019_DAIC_WOZ/split_audio_labels/mapped_dur_test.csv'

# Load index file
Daic_woz_train = pd.read_csv(train_index_file)
Daic_woz_dev = pd.read_csv(dev_index_file)
Daic_woz_test = pd.read_csv(test_index_file)

# Setting the index to 'Wav-path' column and converts the index to string data type.
Daic_woz_train.set_index('Wav-path', inplace=True)
Daic_woz_train.index = Daic_woz_train.index.astype(str)
Daic_woz_dev.set_index('Wav-path', inplace=True)
Daic_woz_dev.index = Daic_woz_dev.index.astype(str)
Daic_woz_test.set_index('Wav-path', inplace=True)
Daic_woz_test.index = Daic_woz_test.index.astype(str)

# Set the output files' (i.e. PDEM embeddings' files) root directory
cache_root = '/home/legalalien/Documents/Jiawei/EDAIC_WOZ_feature'

# Step 3: Extracting PDEM embeddings' using PDEM model for each wav (indicated by the Daic_woz_dev.index loading from the index file).
dev_features_w2v2 = hidden_states.process_index(
    Daic_woz_dev.index,
    root='',
    # The output files will save to the below cache_root directory. The file name will be a random number ending with '.pkl'
    cache_root=audeer.path(cache_root, 'pdem_dev'),
)
# Because the file name is random number, so here we change it to a readable name.
PDEM_dir_path = os.path.join(cache_root, 'pdem_dev')
files = os.listdir(PDEM_dir_path)
for file in files:
    if file.endswith('.pkl'):
        file_path = os.path.join(PDEM_dir_path, file)
        rename_file(file_path,'pdem_dev.pkl')

# The process below is same as above.
train_features_w2v2 = hidden_states.process_index(
    Daic_woz_train.index,
    root='',
    cache_root=audeer.path(cache_root, 'pdem_train'),
)
PDEM_dir_path = os.path.join(cache_root, 'pdem_train')
files = os.listdir(PDEM_dir_path)
for file in files:
    if file.endswith('.pkl'):
        file_path = os.path.join(PDEM_dir_path, file)
        rename_file(file_path,'pdem_train.pkl')


test_features_w2v2 = hidden_states.process_index(
    Daic_woz_test.index,
    root='',
    cache_root=audeer.path(cache_root, 'pdem_test'),
)
PDEM_dir_path = os.path.join(cache_root, 'pdem_test')
files = os.listdir(PDEM_dir_path)
for file in files:
    if file.endswith('.pkl'):
        file_path = os.path.join(PDEM_dir_path, file)
        rename_file(file_path,'pdem_test.pkl')