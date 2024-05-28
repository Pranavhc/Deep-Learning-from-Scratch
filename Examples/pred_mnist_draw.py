import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
from scipy.ndimage import zoom

from nn.utils import load_object

# the model
clf = load_object('Examples/models/mnist_clf.pkl') 

def predict(X):
    return np.argmax(clf.predict(X))

# github copilot wrote this one
def capture_canvas(canvas):
    # Get the canvas's location on the screen
    x = app.winfo_rootx() + canvas.winfo_x()
    y = app.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # removing the border of the canvas
    border = 2
    x += border; y += border
    x1 -= border; y1 -= border

    # Capture the screen content in the canvas's area
    image = ImageGrab.grab(bbox=(x, y, x1, y1))

    # Convert the image into a NumPy array
    pixels = np.array(image)
    
    return pixels

app = tk.Tk()
app.geometry("280x300")

# drawing mouse pointers
def get_x_y(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y), fill='white', width=10)
    last_x, last_y = event.x, event.y

    predict_d()

# erasing mouse pointers
def get_x_y_e(event):
    global last_x_e, last_y_e
    last_x_e, last_y_e = event.x, event.y

def erase(event):
    global last_x_e, last_y_e
    canvas.create_line((last_x_e, last_y_e, event.x, event.y), fill='black', width=15)
    last_x_e, last_y_e = event.x, event.y

    predict_d()

def save_canvas():
    pixels = capture_canvas(canvas)
    image = Image.fromarray(pixels)

    image = image.resize((28, 28))
    image.save('Examples/canvas.png')

# predict drawn digit
def predict_d():
    pixels = capture_canvas(canvas)
    
    # Convert to grayscale (NTSC conversion formula)
    pixels = np.dot(pixels[...,:3], [0.2989, 0.5870, 0.1140])

    # Resize using scipy.ndimage.zoom
    pixels = zoom(pixels, (28 / pixels.shape[0], 28 / pixels.shape[1]))

    pixels = pixels.reshape(1, 784)
    pixels = pixels.astype('float32') / 255

    prediction = predict(pixels)
    prediction_label.config(text=f"Prediction: {prediction}")

""" Unoptimized version of above the function.
    Leaving it here for better understanding. """
def predict_d_unoptimized():
    pixels = capture_canvas(canvas)

    image = Image.fromarray(pixels)
    image = image.resize((28, 28)).convert('L') # Convert to grayscale
    pixels = np.array(image).reshape(1, 784)
    pixels = pixels.astype('float32') / 255

    prediction = predict(pixels)
    prediction_label.config(text=f"Prediction: {prediction}")

def clear_canvas():
    canvas.delete("all")

canvas = tk.Canvas(app, bg='black')
canvas.pack(anchor='nw', fill='both', expand=True)

canvas.bind('<Button-1>', get_x_y)
canvas.bind('<Button-3>', get_x_y_e)
canvas.bind('<B1-Motion>', draw)
canvas.bind('<B3-Motion>', erase)

clear_button = tk.Button(app, text="Clear", command=clear_canvas)
clear_button.pack(side=tk.LEFT)

save_img = tk.Button(app, text="Save image", command=save_canvas)
save_img.pack(side=tk.LEFT)

prediction_label = tk.Label(app, text=" Prediction: ", font=('Arial', 14))
prediction_label.pack(side=tk.LEFT)

app.mainloop()