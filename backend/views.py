# views.py

from django.http import JsonResponse
from keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
from rest_framework.decorators import api_view
from keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input

# Load the skin analysis model
model = load_model('C:/Users/IT TECH/Desktop/facerecognition/Face_recognition/backend/skin.h5', custom_objects={'SparseCategoricalCrossentropy': SparseCategoricalCrossentropy()})

@api_view(['POST'])
def analyze_skin_api(request):
    if request.method == 'POST':
        try:
            # Receive image data from the request
            image_data = request.FILES['image'].read()

            # Pass the image data directly to the preprocess_image function
            processed_image = preprocess_image(image_data)

            # Perform inference using the loaded model
            result = model.predict(processed_image)

            # Return the result
            return JsonResponse({'result': result.tolist()})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)

def preprocess_image(image_data):
    """
    Preprocesses the input image.

    Parameters:
    - image_data: Bytes representing the input image.

    Returns:
    - processed_image: NumPy array representing the preprocessed image.
    """
    # Convert bytes to PIL Image object
    img = Image.open(BytesIO(image_data))

    # Resize the image to match the input size expected by the model
    target_size = (224, 224)  # Adjust according to your model's input size
    img = img.resize(target_size)

    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image (normalize pixel values, etc.)
    processed_image = preprocess_input(img_array)

    return processed_image
