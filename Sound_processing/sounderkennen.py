import glob
import os
import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import mel_spec_calculator as msc
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalMaxPooling2D, BatchNormalization
from D3Modularised import *

model = load_model('full_model_3_new.keras')
print('Model Loaded!')
data = training_data()

model.summary()

