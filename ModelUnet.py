import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D,Conv2D,Dense,BatchNormalization,concatenate,Input,Dropout,Maximum,Activation,Dense,Flatten,UpSampling2D,Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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


# Dice Coeff(F1 Score) = 2* area overlap / Total number of pixels
# The Brain tumor segmentation problem exhibits severe class imbalance where
# the healthy voxels comprise 98% of total voxels,0.18% belongs to necrosis ,1.1% to edema and non-enhanced and 0.38% to enhanced tumor
# Results are presented by the tool mainly in the
# form of well-known Dice Score, Sensitivity (true positive rate) and Specificity (true negative rate) for three main tumor regions;
# whole tumor (all tumor components), core tumor (all tumor components except edema) and active tumor (only active cells).
# Review of MRI-based brain tumor image segmentation using deep learning methods

# The Dice scores obtained for the validation set for whole tumor (WT :NCR/NE +ET +ED), tumor core (TC:NCR/NET +ET), and enhancing tumor (ET) are 0.90216, 0.87247, and 0.82445.
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6993215/
 # 2TP/(2TP+FP+FN)
def dice(y_true, y_pred):
    #computes the dice score on two tensors

    sum_p=K.sum(y_pred)
    sum_r=K.sum(y_true)
    # sum_pr --- TP
    sum_pr=K.sum(y_true * y_pred)
    dice_numerator =2*sum_pr
    dice_denominator =sum_r+sum_p
    dice_score =(dice_numerator+K.epsilon() )/(dice_denominator+K.epsilon())
    return dice_score

def dice_coef_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)


def dice_whole_metric(y_true, y_pred):
    #computes the dice for the whole tumor

    y_true_f = K.reshape(y_true,shape=(-1,4))
    y_pred_f = K.reshape(y_pred,shape=(-1,4))

    # for i in range(6635520):
    # #     if(((y_true_f[i,:]) != np.array([1.,0.,0.,0.])).all()):
    #         print("index, y_true_f values",i,"---------",y_true_f[i,:])

    print(y_true_f.shape, y_pred_f.shape)
    print(y_pred_f[1193599, :])

    y_whole=y_true_f[:,1:]
    p_whole=y_pred_f[:,1:]

    print(y_whole.shape,p_whole.shape)

    dice_whole=dice(y_whole,p_whole)
    return dice_whole

def dice_en_metric(y_true,y_pred):
    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))
    y_enh = y_true_f[:, -1]
    p_enh = y_pred_f[:, -1]
    dice_en = dice(y_enh, p_enh)
    return dice_en


def dice_core_metric(y_true, y_pred):
    ##computes the dice for the core region

    y_true_f = y_true.reshape(-1, 4)
    y_pred_f = y_pred.reshape(-1, 4)

    # workaround for tf
    # y_core=K.sum(tf.gather(y_true_f, [1,3]))
    # p_core=K.sum(tf.gather(y_pred_f, [1,3]))
    y_core=y_true_f[:,[1,3]]
    p_core=y_pred_f[:,[1,3]]
    print("shape y core p core", y_core.shape, p_core.shape)
    dice_core = dice(y_core, p_core)
    return dice_core


#
model.compile(optimizer=Adam(lr=1e-5),loss=dice_coef_loss,metrics=[dice])
# model.load_weights('./Model Checkpoints/weights.hdf5')
# checkpointer = callbacks.ModelCheckpoint(filepath = './Model Checkpoints/weights.hdf5',save_best_only=True)
# training_log = callbacks.TensorBoard(log_dir='./Model_logs')

# history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=32,epochs=16,callbacks=[checkpointer],shuffle=True)
history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),batch_size=16,epochs=16,shuffle=True)


# Confusion matrix for test cases
X_test = np.load('./Validation Data/X_val4.npy')
Y_test = np.load('./Validation Data/Y_val1.npy')
Y_pre = np.load('./Prediction Data/Y_pre_val_with_normalise.npy')
print("Unique values for Y_pre and Y_test", np.unique(Y_pre),np.unique(Y_test))
# Unique values for Y_pre and Y_teszt [0 1 2 3] [0. 1.]


print("X_test.shape","Y_test.shape","Y_pre.shape",X_test.shape,Y_test.shape,Y_pre.shape)
# X_test.shape Y_test.shape Y_pre.shape (1530, 192, 192, 4) (1530, 192, 192, 4) (1530, 192, 192)
y_preRes=Y_pre.reshape(-1)
y_testRes=Y_test.reshape(-1)
print("-------------------------------------------------------------------------")
print("After converting 1d array reshape y_testRes.shape,y_preRes.shape",y_testRes.shape,y_preRes.shape)
# After converting 1d array reshape y_testRes.shape,y_preRes.shape (26542080,) (26542080,)
CM_Y_pre= confusion_matrix(y_testRes,y_preRes)
print("confusion matrix Validation :")
print(CM_Y_pre)

#code from: https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
cf_matrix = confusion_matrix(y_testRes, y_preRes)
#figure1 = sns.heatmap(cf_matrix, annot=True)
#plt.savefig('confusion_matrix_figure1')
print("cf_matrix shape",cf_matrix.shape)

# figure2 = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', cbar=False)
cf_matirx_classpercent = np.zeros(shape=(4,4))
for j in range(4):
    for i in range(4):
        cf_matirx_classpercent[j][i]=cf_matrix[j][i]/ (cf_matrix[j][0]+cf_matrix[j][1]+cf_matrix[j][2]+cf_matrix[j][3])
        print(cf_matirx_classpercent[j][i])

print("cf_matirx_classpercent.shape",np.shape(cf_matirx_classpercent))
# data = np.asarray(cf_matirx_classpercent).reshape(17,1)
figure2 = sns.heatmap(cf_matirx_classpercent,cmap='Blues',annot=True, fmt='.2%', cbar=False)
plt.xlabel("Predicted values")
plt.ylabel("Ground truth values")
plt.savefig('confusion_matrix_percent1')


Y_pre = Y_pre.astype(np.uint8)
Y_test = Y_test.astype(np.uint8)

y_pre = Y_pre.reshape(-1)
y_test = Y_test.reshape(-1)

print("Printing shape before one hot",y_pre.shape,y_test.shape)


# Encoding one hot
Y_pre = to_categorical(Y_pre)
Y_test = to_categorical(Y_test)
print("While calculating Whole Tumour Y_test.shape","Y_pre.shape",X_test.shape,Y_test.shape,Y_pre.shape)
Dice_Whole = dice_whole_metric(Y_test,Y_pre)
Dice_Core = dice_core_metric(Y_test,Y_pre)
Dice_Enhanced = dice_en_metric(Y_test,Y_pre)
print("Dice_Whole value for Validate",Dice_Whole)
print("Dice_Core value for Validate",Dice_Core)
print("Dice_Enhanced value for Validate",Dice_Enhanced)


