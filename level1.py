###########################################
#################LeafLens################
###########################################

import cv2
def preprocess_img(image):
    rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(rgb,(224,224))
    scaled_rgb = resized_rgb.astype("float32")/255.0
    return scaled_rgb

image_path = r"D:\Download\leaf-image-procession\leaf-image-procession\archive\plantvillage dataset\color\Apple___Apple_scab\2c89ceaf-748c-4371-80d0-d01855f04a92___FREC_Scab 2962.JPG"
image = cv2.imread(image_path)
if image is None:
    print("Image could not be loaded")
else:
    print("Image loaded successfully")
    scaled_rgb = preprocess_img(image)
    
    print("Final shape: ",scaled_rgb.shape)
    print("Data type:",scaled_rgb.dtype)
    print("Value range:",scaled_rgb.min(),scaled_rgb.max())
    cv2.imshow("My leaf:",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()