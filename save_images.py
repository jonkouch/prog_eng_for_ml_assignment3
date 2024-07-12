from sklearn.datasets import load_digits
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split


digits = load_digits()

# save a few images from the digits dataset test set to a directory
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# save the first 10 test images
for i, (image, target) in enumerate(zip(X_test[:10], y_test[:10])):
    # save the image
    img = Image.fromarray(image.reshape(8, 8).astype('uint8'))
    dir = 'input_images'
    img.save(f'{dir}/test_image_{i}.png')