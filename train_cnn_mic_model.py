import re
import os
import sys
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import keras
# import torch
from time import time
import pubchempy as pcp
import tensorflow as tf
from rdkit.Chem import MolFromSmiles
import tensorflow.keras.backend as K
from molvecgen import SmilesVectorizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

random.seed(1)
np.random.seed(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

gpus = tf.config.experimental.list_physical_devices("GPU")
for i in range(len(gpus)):
    tf.config.experimental.set_memory_growth(gpus[i], True)
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1", "/GPU:2", "/GPU:3", "/GPU:4", "/GPU:5", "/GPU:6", "/GPU:7"])

from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)

keras = tf.compat.v1.keras

########################################################################################

anti_cid_dict = {'Amikacin':37768,'Ampicillin':6249,'Ampicillin/Sulbactam':74331798,'Aztreonam':5742832,'Cefazolin':33255,'Cefepime':5479537,'Cefoxitin':441199,'Ceftazidime':5481173,'Ceftriaxone':5479530,'Cefuroxime sodium':23670318,
                'Ciprofloxacin':2764,'Gentamicin':3467,'Imipenem':104838,'Levofloxacin':149096,'Meropenem':441130,'Nitrofurantoin':6604200,'Piperacillin/Tazobactam':9918881,'Tetracycline':54675776,'Tobramycin':36294,'Trimethoprim/Sulfamethoxazole':358641}

kmc_dir = sys.argv[1] + '/'
ten_mers_index = sys.argv[2]
train_map = sys.argv[3]
val_map = sys.argv[4]
test_map = sys.argv[5]
model_rep = sys.argv[6]
val_result = sys.argv[7]
test_result = sys.argv[8]

########################################################################################

def getAntiMat(anti_cid_dict):
    anti_smiles_dict = {}
    for key in anti_cid_dict:
        anti_smiles_dict[key] = pcp.Compound.from_cid(anti_cid_dict[key]).isomeric_smiles
    anti_mol_dict = {}
    for key in anti_smiles_dict:
        anti_mol_dict[key] = MolFromSmiles(anti_smiles_dict[key])

    anti_mat_dict = {}
    encoder = SmilesVectorizer()
    for key in anti_mol_dict:
        mol = [anti_mol_dict[key]]
        anti_mat_dict[key] = encoder.transform(mol)[0]
        for i in range(8):
            anti_mat_dict[key] = np.vstack((anti_mat_dict[key], anti_mat_dict[key]))
        anti_mat_dict[key] = anti_mat_dict[key][11071:]
    return anti_mat_dict

def convertMIC(s):
    new_s = re.sub('\>([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))*2), s)
    new_s = re.sub('\<([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))/2), new_s)
    new_s = re.sub('\<=([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))), new_s)
    new_s = re.sub('^([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))), new_s)
    return float(new_s)

def getListData(data_file):
    df = pd.read_csv(data_file)
    NB_SAMPLES = len(df)
    df['MIC'] = df['Actual MIC'].apply(convertMIC)
    lst_data = []
    for index, row in df.iterrows():
       lst_data.append({'PATRIC ID':row['PATRIC ID'], 'MIC': row['MIC'], 'ANTI': row['Antibiotic']})
    random.shuffle(lst_data)
    return lst_data

def convertKmers(filename, anti):
    BASE_NUM = 1.5529
    features = np.zeros((22209, 20), dtype=np.float32)
    anti_mat = anti_mat_dict[anti]
    with open(kmc_dir + '{:0.5f}'.format(filename) , 'r') as data_file:
        for index, line in enumerate(data_file.read().splitlines()):
            kmer = line.split('\t')[0].upper()
            if kmer in kmers_index:
                kmer_counts = int(line.split("\t")[-1])
                if kmer_counts == 1:
                    features[kmers_index[kmer]][0] = 1
                else:
                    log_num = math.ceil(math.log(kmer_counts, BASE_NUM))
                    features[kmers_index[kmer]][log_num] = 1
    matrix = np.transpose([np.add(features, anti_mat)/2], (1,2,0))
    return matrix

def get_compiled_model():
    with mirrored_strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same', input_shape=(22209,20,1)),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),

            tf.keras.layers.Dense(11, activation='softmax')
        ])

        opt = keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy())
    return model

def train(lstTrain, lstVal, modelRep):    
    training_generator = DataGenerator(lstTrain)
    validation_generator = DataGenerator(lstVal)
    
    # Train model on dataset
    model = get_compiled_model()
    print(model.summary())
    keras.utils.plot_model(model, show_layer_names=True, show_shapes=True, to_file= modelRep+'.png')

    tensorboard = TensorBoard(log_dir=modelRep+'_tensorboard')
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=1, min_lr=1e-10)
      
    history = model.fit(training_generator, 
                    validation_data=validation_generator,
                    epochs=40,
                    use_multiprocessing=True,
                    workers=32, 
                    callbacks=[tensorboard, early_stopping, reduce_lr],
                    verbose=1
                    )
    model.save(modelRep+'.h5')

def test(lstTest, modelRep, fileResult):
    print("\n#### TEST PROCESS ")    
    print("Number of test samples: ", len(lstTest))
    
    model = tf.keras.models.load_model(modelRep+'.h5')
    print(model.summary())
    nbCorrect = 0
    exact = 0
    f_res = open(fileResult, "w")
    f_res.write("PATRIC_ID, ANTI, MIC, Predict MIC")
    cnt = 0
    print()
    print("Testing...")
    for test_sample in tqdm(lstTest):
        cnt+=1
        data_sample = np.asarray([convertKmers(test_sample['PATRIC ID'], test_sample['ANTI'])])
        target_sample = int(math.log(test_sample["MIC"],2))
        res = np.argmax(model.predict(data_sample), axis=-1) - 3
        MIC_res = math.pow(2, res)
        if int(res) == target_sample:
            exact += 1
        else:
            exact += 0
        if abs(res - target_sample) <= 1:
            nbCorrect += 1
        else:
            nbCorrect += 0
        f_res.write("\n" + str(test_sample['PATRIC ID']) + "," + str(test_sample['ANTI']) + "," + str(test_sample["MIC"]) + "," + str(MIC_res))    
    
    print("EXACT TEST ACC: ", exact, " / ", len(lstTest), " = ", exact / len(lstTest))
    print("W/I 1-TIER Test ACC: ", nbCorrect, " / ", len(lstTest), " = ", nbCorrect / len(lstTest))
    f_res.write("\nEXACT Test ACC: " + str(exact) + " / " + str(len(lstTest)) + " = " + str( exact / len(lstTest)))
    f_res.write("\nW/I 1-TIER Test ACC: " + str(nbCorrect) + " / " + str(len(lstTest)) + " = " + str( nbCorrect / len(lstTest)))
    f_res.close()

########################################################################################

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, lst_data, batch_size=32, dim=(22209,20), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.lst_data = lst_data
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.lst_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        lst_data_temp = [self.lst_data[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(lst_data_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.lst_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, lst_data_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 11))

        # Generate data
        for i, item in enumerate(lst_data_temp):
            # Store sample
            X[i,] = convertKmers(item['PATRIC ID'], item['ANTI'])
            # Store class
            y[i] = tf.keras.utils.to_categorical(math.log(item['MIC'], 2) + 3, num_classes=11)

        return X, y

########################################################################################

if __name__ == "__main__":
    anti_mat_dict = getAntiMat(anti_cid_dict)
    
    with open(ten_mers_index) as f:
        lines = f.readlines()
    kmers_index = {}
    for line in lines:
        kmers_index[line.split('\t')[0].upper()] = int(line.split('\t')[-1])
    print("kmers_index :", len(kmers_index))

    lstTrain = getListData(train_map)
    lstVal = getListData(val_map)
    lstTest = getListData(test_map)
    print("lstTrain: ", len(lstTrain))
    print("lstVal: ", len(lstVal))
    print("lstTest: ", len(lstTest))
    print("total: ", len(lstTrain) + len(lstVal) + len(lstTest))

    train(lstTrain, lstVal, model_rep)
    test(lstVal, modelRep, val_result)
    test(lstTest, modelRep, test_result)
    