import numpy as np

def normalize(img,bottom = 99,top = 1):

    b = np.percentile(img,bottom)
    # print("bottom",b)

    t = np.percentile(img,top)
    # print("top",t)

    clipImage = np.clip (img,t,b)


    if np.std(clipImage) == 0:
        return clipImage
    else:
        # print("Mean of image",np.mean(clipImage))
        # print("Std of image",np.std(clipImage))
        # print("Min of image", np.min(clipImage))
        # print("Max of image", np.max(clipImage))
        normImage = (clipImage - np.mean(clipImage)) / np.std (clipImage)
        return normImage

#     count images with 0 intensity
#     Remove all voxels with intensity values 0

def transposeData(data):
    data = np.transpose(data, (0, 2, 3, 4, 1))
    return data

def sliceCrop(data,gt):
    # As all slices does not show tumour region so only mid-portion i.e. 30th - 120th slice was taken to create final data
    # each data is also cropped to centre with final dimension of (N1,192,192,4)
    data = data[:,30:120,30:222,30:222,:].reshape([-1,192,192,4])
    gt = gt[:,30:120,30:222,30:222].reshape([-1,192,192,1])
    return data,gt

def groundTruth4to3(gt):
    #  GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    gt[np.where(gt==4)]=3   #converting ground truth value of 4 to 3 to do one hot encoding
    return gt
    # (Consider value 3 in results in output as class 4)