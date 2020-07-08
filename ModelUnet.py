import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Dense,BatchNormalization,concatenate,Input,Dropout,Maximum,Activation,Dense,Flatten,UpSampling2D,Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.callbacks as callbacks
# import keras.initializers as initializers
# from keras.callbacks import Callback
# from keras import regularizers
from sklearn.metrics import confusion_matrix

X_train = np.load('./Training Data/X_train4.npy')
Y_train = np.load('./Training Data/Y_train4.npy')

X_val = np.load('./Validation Data/X_val4.npy')
Y_val = np.load('./Validation Data/Y_val4.npy')
Y_val1 = np.load('./Validation Data/Y_val1.npy')



# X_train, Y_train shape (4500, 192, 192, 4) (4500, 192, 192, 4)
# X_val, Y_val shape (1530, 192, 192, 4) (1530, 192, 192, 4)


input_ = Input(shape=(192,192,4),name='input')

block1_conv1 = Conv2D(64,3,padding='same',activation='relu',name='block1_conv1')(input_)
block1_conv2 = Conv2D(64,3,padding='same',activation='relu',name='block1_conv2')(block1_conv1)
block1_norm = BatchNormalization(name='block1_batch_norm')(block1_conv2)
block1_pool = MaxPooling2D(name='block1_pool')(block1_norm)

block2_conv1 = Conv2D(128,3,padding='same',activation='relu',name='block2_conv1')(block1_pool)
block2_conv2 = Conv2D(128,3,padding='same',activation='relu',name='block2_conv2')(block2_conv1)
block2_norm = BatchNormalization(name='block2_batch_norm')(block2_conv2)
block2_pool = MaxPooling2D(name='block2_pool')(block2_norm)

encoder_dropout_1 = Dropout(0.2,name='encoder_dropout_1')(block2_pool)

block3_conv1 = Conv2D(256,3,padding='same',activation='relu',name='block3_conv1')(encoder_dropout_1)
block3_conv2 = Conv2D(256,3,padding='same',activation='relu',name='block3_conv2')(block3_conv1)
block3_norm = BatchNormalization(name='block3_batch_norm')(block3_conv2)
block3_pool = MaxPooling2D(name='block3_pool')(block3_norm)

block4_conv1 = Conv2D(512,3,padding='same',activation='relu',name='block4_conv1')(block3_pool)
block4_conv2 = Conv2D(512,3,padding='same',activation='relu',name='block4_conv2')(block4_conv1)
block4_norm = BatchNormalization(name='block4_batch_norm')(block4_conv2)
block4_pool = MaxPooling2D(name='block4_pool')(block4_norm)
################### Encoder end ######################

block5_conv1 = Conv2D(1024,3,padding='same',activation='relu',name='block5_conv1')(block4_pool)
# encoder_dropout_2 = Dropout(0.2,name='encoder_dropout_2')(block5_conv1)

########### Decoder ################

up_pool1 = Conv2DTranspose(1024,3,strides = (2, 2),padding='same',activation='relu',name='up_pool1')(block5_conv1)
merged_block1 = concatenate([block4_norm,up_pool1],name='merged_block1')
decod_block1_conv1 = Conv2D(512,3, padding = 'same', activation='relu',name='decod_block1_conv1')(merged_block1)

up_pool2 = Conv2DTranspose(512,3,strides = (2, 2),padding='same',activation='relu',name='up_pool2')(decod_block1_conv1)
merged_block2 = concatenate([block3_norm,up_pool2],name='merged_block2')
decod_block2_conv1 = Conv2D(256,3,padding = 'same',activation='relu',name='decod_block2_conv1')(merged_block2)

decoder_dropout_1 = Dropout(0.2,name='decoder_dropout_1')(decod_block2_conv1)

up_pool3 = Conv2DTranspose(256,3,strides = (2, 2),padding='same',activation='relu',name='up_pool3')(decoder_dropout_1)
merged_block3 = concatenate([block2_norm,up_pool3],name='merged_block3')
decod_block3_conv1 = Conv2D(128,3,padding = 'same',activation='relu',name='decod_block3_conv1')(merged_block3)

up_pool4 = Conv2DTranspose(128,3,strides = (2, 2),padding='same',activation='relu',name='up_pool4')(decod_block3_conv1)
merged_block4 = concatenate([block1_norm,up_pool4],name='merged_block4')
decod_block4_conv1 = Conv2D(64,3,padding = 'same',activation='relu',name='decod_block4_conv1')(merged_block4)
############ Decoder End ######################################

# decoder_dropout_2 = Dropout(0.2,name='decoder_dropout_2')(decod_block4_conv1)

pre_output = Conv2D(64,1,padding = 'same',activation='relu',name='pre_output')(decod_block4_conv1)

output = Conv2D(4,1,padding='same',activation='softmax',name='output')(pre_output)

model = Model(inputs = input_, outputs = output)
print(model.summary())



# from keras.utils import  plot_model
# plot_model(model,to_file='unet.png',show_shapes=True)

# print("X_test.shape",X_test.shape)
# print("Y_test.shape",Y_test.shape)

# IOU
# categorical_crossentropy
# Dice Coeff(F1 Score) = 2* area overlap / Total number of pixels

from tensorflow.keras import backend as K
def dice_coef(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection) / (K.sum(K.square(y_true),axis=-1) + K.sum(K.square(y_pred),axis=-1) + epsilon)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef_loss,metrics=[dice_coef])
# model.load_weights('./Model Checkpoints/weights.hdf5')
# checkpointer = callbacks.ModelCheckpoint(filepath = './Model Checkpoints/weights.hdf5',save_best_only=True)
# training_log = callbacks.TensorBoard(log_dir='./Model_logs')

# history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=32,epochs=16,callbacks=[checkpointer],shuffle=True)
history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=16,epochs=10,shuffle=True)

# summarize history for loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('dice coef loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model score')
plt.ylabel('dice coef')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Evaluation of Model

X_test = np.load('./Test Data/X_test4.npy')
Y_test = np.load('./Test Data/Y_test4.npy')

# Confusion matrix for test cases
Y_pre = np.argmax(model.predict(X_test),axis=-1)
print("Unique values for Y_pre and Y_test", np.unique(Y_pre),np.unique(Y_test))
# Unique values for Y_pre and Y_test [0 1 2 3] [0 1 2 3]
print("X_test.shape","Y_test.shape","Y_pre.shape",X_test.shape,Y_test.shape,Y_pre.shape)
# X_test.shape,Y_test.shape,Y_pre.shape (720, 192, 192, 4),(720, 192, 192, 1) (720, 192, 192)
y_preRes=Y_pre.reshape(-1)
y_testRes=Y_test.reshape(-1)
print("-------------------------------------------------------------------------")
print("After converting 1d array reshape y_testRes.shape,y_preRes.shape",y_testRes.shape,y_preRes.shape)
# After converting 1d array reshape y_testRes.shape,y_preRes.shape (26542080,) (26542080,)
print("confusion matrix Test :", confusion_matrix(y_testRes,y_preRes))


# For printing of Test images
Y_pre=Y_pre.reshape(-1,192,192,1)
print("X_test.shape","Y_test.shape","Y_pre.shape",Y_test.shape,X_test.shape,Y_pre.shape)
# X_test.shape Y_test.shape Y_pre.shape (720, 192, 192, 1) (720, 192, 192, 4) (720, 192, 192, 1)
for i in range(470,475):
  print('X_test '+ str(i))
  plt.imshow(X_test[i,:,:,2])
  plt.savefig('X_test '+ str(i))
  plt.show()
  plt.imshow(Y_pre[i,:,:,0])
  plt.savefig('Y_pre '+ str(i))
  plt.show()
  plt.imshow(Y_test[i,:,:,0])
  plt.savefig('Y_test ' + str(i))
  plt.show()


# confusion matrix for val
Y_val_pre = np.argmax(model.predict(X_val),axis=-1)
print("Unique values for Y_val_pre and Y_val", np.unique(Y_val_pre),np.unique(Y_val))
# Unique values for Y_val_pre and Y_val [0 1 2 3] [0. 1.]
print("Y_val1.shape,Y_val_pre.shape",Y_val1.shape,Y_val_pre.shape)
# Y_val1.shape,Y_val_pre.shape (1530, 192, 192, 1) (1530, 192, 192)
# Changing the array size for confusion matrix
y_val_preRes=Y_val_pre.ravel()
# original Y val instead of the one hot Y val
Y_val1Res=Y_val1.ravel()
print("After converting 1d array reshape Y_val.shape,Y_val_pre.shape",Y_val1Res.shape,y_val_preRes.shape)
# After converting 1d array reshape Y_val.shape,Y_val_pre.shape (56401920,) (56401920,)
print("confusion matrix Validation :", confusion_matrix(Y_val1Res,y_val_preRes))


# For printing of Validation images
Y_val_pre=Y_val_pre.reshape(-1,192,192,1)
print("X_val.shape","Y_val.shape","Y_val_pre.shape",X_val.shape,Y_val.shape,Y_val_pre.shape)
# X_val.shape Y_val.shape Y_val_pre.shape (1530, 192, 192, 4) (1530, 192, 192, 4) (1530, 192, 192, 1)
for i in range(470,475):
  print('X_val '+ str(i))
  plt.imshow(X_val[i,:,:,2])
  plt.savefig('X_val '+ str(i))
  plt.show()
  plt.imshow(Y_val_pre[i,:,:,0])
  plt.savefig('Y_val_pre '+ str(i))
  plt.show()
  plt.imshow(Y_val1[i,:,:,0])
  plt.savefig('Y_val1' + str(i))
  plt.show()


from tensorflow.keras.utils import to_categorical
Y_test_encod = to_categorical(Y_test)

print("Model evaluation",model.evaluate(X_test,Y_test_encod,verbose=0))

model.save('./Saved Models/model_new.h5',overwrite=True)

print("ModelSaved Successfully")