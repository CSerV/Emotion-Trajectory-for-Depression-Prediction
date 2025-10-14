import audeer
import audonnx
import numpy as np
import pandas as pd
import audinterface
import os


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
model_cache_root = audeer.mkdir('pdem_cache')
model_root = audeer.mkdir('pdem_model')

# Step 1: Download the PDEM model to the model_root dir and Load the PDEM model
archive_path = audeer.download_url(url, model_cache_root, verbose=True)
audeer.extract_archive(archive_path, model_root)

# Load model from model_root using audonnx package.
model = audonnx.load(model_root)
# Set audio sampling rate. 
# I think the sampling rate of DAIC-WOZ audio data is 16000 indeed, so you don't need to change this setting.
sampling_rate = 16000

# Define model interface, output is valence, arousal and donimance scores. This format is fixed defined by audinterface pacakage.
# You can adjust the parameter num_workers, the bigger the faster, as long as your computer can tolerate.
logits = audinterface.Feature(
    model.labels('logits'),
    process_func=model,
    process_func_args={
        'outputs': 'logits',
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

# Set the output files' (i.e. VAD scores' files) root directory
cache_root = '/home/legalalien/Documents/Jiawei/EDAIC_WOZ_feature'

# Step 3: Extracting VAD scores using PDEM model for each wav (indicated by the Daic_woz_dev.index loading from the index file).
dev_features_w2v2 = logits.process_index(
    Daic_woz_dev.index,
    root='',
    # The output files will save to the below cache_root directory. The file name will be a random number ending with '.pkl'
    cache_root=audeer.path(cache_root, 'vad_dev'),
)
# Because the file name is random number, so here we change it to a readable name.
VAD_dir_path = os.path.join(cache_root, 'vad_dev')
files = os.listdir(VAD_dir_path)
for file in files:
    if file.endswith('.pkl'):
        file_path = os.path.join(VAD_dir_path, file)
        rename_file(file_path,'vad_dev.pkl')

# The process below is same as above.
train_features_w2v2 = logits.process_index(
    Daic_woz_train.index,
    root='',
    cache_root=audeer.path(cache_root, 'vad_train'),
)
VAD_dir_path = os.path.join(cache_root, 'vad_train')
files = os.listdir(VAD_dir_path)
for file in files:
    if file.endswith('.pkl'):
        file_path = os.path.join(VAD_dir_path, file)
        rename_file(file_path,'vad_train.pkl')


test_features_w2v2 = logits.process_index(
    Daic_woz_test.index,
    root='',
    cache_root=audeer.path(cache_root, 'vad_test'),
)
VAD_dir_path = os.path.join(cache_root, 'vad_test')
files = os.listdir(VAD_dir_path)
for file in files:
    if file.endswith('.pkl'):
        file_path = os.path.join(VAD_dir_path, file)
        rename_file(file_path,'vad_test.pkl')
