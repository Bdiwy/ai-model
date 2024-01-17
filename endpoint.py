import requests
from PIL import Image
import numpy as np
import tensorflow.lite as tflite

# Load the TensorFlow Lite model
# Model Loading:
# The script loads a TensorFlow Lite model from the specified path using the tflite.Interpreter class

model_path = 'efficientnetb3-EyeDisease-96.22.tflite'
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()




# Image Preprocessing:
# The script defines a function preprocess_image that loads and
# preprocesses an image using the Python Imaging Library (PIL) and 
# NumPy. The image is resized to (224, 224) pixels,
# normalized to values between 0 and 1, and expanded along the first axis.

def preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype(np.float32)




# Inference:
# The script defines a function perform_inference that takes the preprocessed image array,
# performs inference using the loaded TensorFlow Lite model, and returns the predictions as a list.

def perform_inference(image_array):
    # Perform inference
    input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
    input_tensor()[0] = image_array
    interpreter.invoke()
    output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])
    predictions = output_tensor()[0]
    return predictions.tolist()

def send_results(predictions):
    # Send predictions to a specified URL
    url = 'http://example.com/result'
    try:
        response = requests.post(url, json={'predictions': predictions})
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {'error': f'HTTP error: {http_err}'}
    except requests.exceptions.RequestException as req_err:
        return {'error': f'Request error: {req_err}'}
    
if __name__ == '__main__':


# Image URL and Inference:
# The script then specifies three URLs for different images (x, y, and z), 
# retrieves the images using requests.get, preprocesses each image, and performs inference.

    image_url = 'http://127.0.0.1:8000/imgs/eye_photo/hero_eye_img.jpg'
    x = 'https://scontent.fcai20-4.fna.fbcdn.net/v/t39.30808-6/419231221_1763815340799609_2932165302945966115_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=3635dc&_nc_eui2=AeEIGoXgONH-C5ytZpYWuvybxU1DSwLzFfvFTUNLAvMV-y_m_SL3BEped8_bsoDUZQ3D4KMmtrYpXlhvO9wonSlD&_nc_ohc=0ACKBzvAb3oAX_Qu0xe&_nc_ht=scontent.fcai20-4.fna&oh=00_AfBu9bSI8iQVZCe0v4qzNKqjHGXXLGBifS4xLVUviho8Hg&oe=65AC037E'  # Replace with the actual image URL
    y ='https://scontent.fcai20-4.fna.fbcdn.net/v/t39.30808-6/417579366_1763826974131779_7624810667180874707_n.jpg?_nc_cat=109&ccb=1-7&_nc_sid=3635dc&_nc_eui2=AeFT_fevAz6iXK7pq85vwPb60DONf7rURrvQM41_utRGuwd-TEoirCbRtwXCvQzSfzhfw2yoZsmJnu0m7s9v_J8m&_nc_ohc=UhCBAdEhhZMAX-8eJpO&_nc_ht=scontent.fcai20-4.fna&oh=00_AfBtNqMhsBFDoNeh_zJQWYhPBYMqXUwm04B0gpy3IDPqAg&oe=65AB3C82'
    z ='https://scontent.fcai20-4.fna.fbcdn.net/v/t39.30808-6/417391021_1763828494131627_6430412670170126823_n.jpg?_nc_cat=102&ccb=1-7&_nc_sid=3635dc&_nc_eui2=AeEFglWmTN1w6zxre72O1ICZfT-aj1-aN6x9P5qPX5o3rFBQBWrWL5X_ApMDQXVHXRgomv3pnoSp243ZfVft44OI&_nc_ohc=OJUwr9o9upoAX-DnK4O&_nc_ht=scontent.fcai20-4.fna&oh=00_AfD0Qp23GWLMi12fGtwUhRGqr_kIfVIHqwI3QO26vqFFJg&oe=65AD0D83'
    image_array = preprocess_image(requests.get(image_url, stream=True).raw)

    predictions = perform_inference(image_array)
    # result = send_results(predictions)
    result = predictions
    print('Inference Result:', result)
