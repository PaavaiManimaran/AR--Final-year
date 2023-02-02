import cv2
import numpy as np
import tensorflow as tf

# Load the DeepLab model
model = tf.keras.models.load_model("deeplabv3_mnv2_pascal_trainval/frozen_inference_graph.pb")

def run_deeplab(img):
    # Preprocess the image
    img = cv2.resize(img, (512, 512))
    img = img[np.newaxis, ...].astype("float32") / 255
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # Run the model
    result = model.predict(img)

    # Postprocess the result
    result = result[0]
    result = cv2.resize(result, (img.shape[2], img.shape[1]))
    result = np.argmax(result, axis=-1).astype("uint8")

    # Create a mask with the foreground segmentation
    mask = result == 15

    # Remove the background
    img = img[0] * 255
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[~mask] = [0, 0, 0]

    return img

# Load an example image
img = cv2.imread("example.jpg")

# Run the DeepLab model
result = run_deeplab(img)

# Save the result
cv2.imwrite("result.png", result)


Note * that this code assumes you have the DeepLabv3 model (with MobileNetV2 backbone) pre-trained on the Pascal VOC dataset, and you should replace the path to the model with the path to the one you have.



