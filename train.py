from model.MwAN import MwAN
from prepro.preprocess import process_data
import pickle

data_path="data/"
# vocab_size=process_data(data_path=data_path,threshold=5)
vocab_size = 98745

with open(data_path + 'train.pickle', 'rb') as f:
    train_data = pickle.load(f)
with open(data_path + 'dev.pickle', 'rb') as f:
    dev_data = pickle.load(f)
with open(data_path + 'testa.pickle', 'rb') as f:
    test_data = pickle.load(f)
dev_data = sorted(dev_data, key=lambda x: len(x[1]))

print('train data size {:d}, dev data size {:d}, testa data size {:d}'.format(len(train_data), len(dev_data),len(test_data)))

model=MwAN()
model.build() 

