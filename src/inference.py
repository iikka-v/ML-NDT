
from __future__ import print_function
import sys
import keras

import numpy as np
import matplotlib.pyplot as plt

model_path = sys.argv[1]
data_path  = sys.argv[2]

model = keras.models.load_model(model_path)
rxs = np.fromfile(data_path, dtype=np.uint16 ).astype('float32')
rxs -= rxs.mean()
rxs /= rxs.std()+0.0001
rxs = np.reshape( rxs, (-1,256,256,1), 'C')

predictions = model.predict(rxs)
print(predictions)
