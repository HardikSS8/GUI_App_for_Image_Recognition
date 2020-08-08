# import the necessary built-in libraries of Python
import numpy
import tkinter as tk

from tkinter import filedialog
from tkinter import *

from PIL import ImageTk, Image

from tensorflow.keras.models import model_from_json
from pathlib import Path


# Load the json file that contains the model's structure
f = Path("trained_model.json")
model_structure = f.read_text()

# Recreate the Keras model object from the json data
model = model_from_json(model_structure)

# Re-load the model's trained weights
model.load_weights("trained_weights.h5")

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
image_classes = {
    0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

# Built the GUI

# Initialise the TKinter window with following parameters
top = tk.Tk()

# Window size of 600 * 600
top.geometry('600x600')

# Heading of the TKinter window
top.title('Image Classification GUI')

# Background color of whole window
top.configure(background='#ffffff')

# Heading : Image Classification GUI APP
label = Label(top, background='#ffffff', font=('Comic Sans MS', 25, 'bold italic'))
label.pack(side=LEFT, padx=30, pady=30)
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)

    # Load an image file to test, resizing it to 32x32 pixels (as required by this model)
    image = image.resize((32, 32))

    # Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
    image = numpy.expand_dims(image, axis=0)

    # Convert the image to a numpy array
    image = numpy.array(image)

    # Make a prediction using the model
    # Since we are only testing one image, we only need to check the first result
    pred = model.predict_classes([image])[0]

    # Get the name of the most likely class
    sign = image_classes[pred]
    print(sign)

    # Print the result in GUI with following specifications
    label.configure(background='#dddddd', text="Class of image : " + sign, font=('Comic Sans MS', 18))


# ADD BUTTON : Result
def show_classify_button(file_path):
    classify_b = Button(top, text="Result", command=lambda: classify(file_path), padx=5, pady=5)
    classify_b.configure(background='#dddddd', font=('Comic Sans MS', 14))
    classify_b.place(relx=0.75, rely=0.68)


# Functionality of uploading image from file
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 1.50), (top.winfo_height() / 1.50)))

        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im

        label.configure(text=None)
        show_classify_button(file_path)
    except:
        pass


# Button configuration of : Upload image here
upload = Button(top, text="Upload image here", command=upload_image, padx=10, pady=6)
upload.configure(background='#dddddd', font=('Comic Sans MS', 14))
upload.pack(side=BOTTOM, pady=0)

sign_image.pack(side=BOTTOM, expand=True)

label.pack(side=BOTTOM, expand=True)

# Heading of the GUI Application : Image Classification GUI App
heading = Label(top, background="#f3f3f3", text="Image Classification GUI App", pady=10, font=('Comic Sans MS', 25, 'bold italic'))
heading.configure(background='#f3f3f3')
heading.pack(side=LEFT, padx=30, pady=30)

top.mainloop()
