import numpy as np
from preprocessing import *

data = np.load('./LG_data.npy')
gt = np.load('./LG_gt.npy')

# 70 : 20 : 10
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, gt, test_size=0.10, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25,random_state=42)


print("After split X_train, Y_train shape",X_train.shape, Y_train.shape)
print("After split X_val, Y_val shape",X_val.shape, Y_val.shape)
print("After split X_test, Y_test shape",X_test.shape, Y_test.shape)


# After split X_train, Y_train shape (50, 4, 155, 240, 240) (50, 155, 240, 240)
# After split X_val, Y_val shape (17, 4, 155, 240, 240) (17, 155, 240, 240)
# After split X_test, Y_test shape (8, 4, 155, 240, 240) (8, 155, 240, 240)


X_train=transposeData(X_train)
X_train,Y_train=sliceCrop(X_train,Y_train)

X_val=transposeData(X_val)
X_val,Y_val=sliceCrop(X_val,Y_val)

X_test=transposeData(X_test)
X_test,Y_test=sliceCrop(X_test,Y_test)

Y_train = groundTruth4to3(Y_train)
Y_val =  groundTruth4to3(Y_val)
Y_test =  groundTruth4to3(Y_test)

# for confusion matrix
Y_val1=Y_val

print("Before to_categorical")
print("X_train, Y_train shape",X_train.shape, Y_train.shape)
print("X_val, Y_val shape",X_val.shape, Y_val.shape)
print("X_test, Y_test shape",X_test.shape, Y_test.shape)
print("Y_train unique",np.unique(Y_train))

# If your training data uses classes as numbers, to_categorical will transform those numbers in proper vectors for using with models
from keras.utils import to_categorical
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)

print("Before saving to files data shape")
print("X_train, Y_train shape",X_train.shape, Y_train.shape)
print("X_val, Y_val shape",X_val.shape, Y_val.shape)
print("X_test, Y_test shape",X_test.shape, Y_test.shape)
print("Y_train unique",np.unique(Y_train))

# np.save('./Training Data/X_train4.npy',X_train)
# np.save('./Training Data/Y_train4.npy',Y_train)
# np.save('./Validation Data/X_val4.npy',X_val)
# np.save('./Validation Data/Y_val4.npy',Y_val)
# np.save('./Test Data/X_test4.npy',X_test)
# np.save('./Test Data/Y_test4.npy',Y_test)

np.save('./Validation Data/Y_val1.npy',Y_val1)

print("Data saved successfully")