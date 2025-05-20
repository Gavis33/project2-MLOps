import os 
import pickle

pkl_file_path = '../models/vectorizer.pkl'

if os.path.exists(pkl_file_path):
    try:
        vectorizer = pickle.load(open(pkl_file_path, 'rb'))
        print(f'File found at {pkl_file_path} and loaded successfully')
    except Exception as e:
        print(f'Error occurred while loading the model: {e}')

else:
    print(f'File not found at {pkl_file_path}')