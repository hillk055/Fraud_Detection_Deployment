import pandas as pd 
import torch 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('creditcard.csv')
x, y = df.drop('Class', axis=1), df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, stratify=y, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, stratify=y_test, random_state=42)

# Fit scaler to training data only

scaler = StandardScaler()
scaler.fit(x_train)

scaled_x = [scaler.transform(x) for x in [x_train, x_val, x_test]]

x_train_tensor, x_val_tensor, x_test_tensor = [torch.from_numpy(x) for x in scaled_x]

print(x_train_tensor.dtype)