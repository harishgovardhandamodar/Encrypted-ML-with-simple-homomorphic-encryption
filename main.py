import pandas as pd
from utils import dataload
from model import vanillaModel, homomorphicEncryptionModel
import os
current_dir = os.getcwd()
cci_data = pd.read_csv(current_dir+"/uci_cci.csv")

input_cols = list(cci_data.columns)[:-1]
output_cols = list(cci_data.columns)[-1]

x_data, y_data, x_data_shortened, y_data_shortened, X_enc, y_enc, H_enc = dataload(cci_data, input_cols, output_cols)

# https://www.cs.cmu.edu/~rjhall/JOS_revised_May_31a.pdf
# source for use of orthogonal matrices and invertible matrices transformation for homomorophic encryption

# Test case 1: Vanilla model (benchmark credit rating/scoring model for default prediction)
vanillaModel(x_data, y_data)
# Test case 2: Homomorphic encryption model running on encrypted data
homomorphicEncryptionModel(X_enc, y_enc, x_data, H_enc)