import os, re

import numpy as np
import cv2

from keras.models import load_model
import tkinter as tk

from tkinter.filedialog import askopenfilenames, askdirectory
# from tkfilebrowser import askopenfilenames, askopendirname

from mtcnn.mtcnn import MTCNN

from PIL import Image, ImageTk

detector = MTCNN()
model = load_model("emotion_detector_models/model_v6_23.hdf5")


class GUI(tk.Frame):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.pack(anchor=tk.NW)
        self.canvas = tk.Canvas(self, width=width + 2, height=height + 2)

        self.canvas.pack()

    def draw(self, x1, y1, x2, y2, text):
        line = (x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)

        self.canvas.create_line(line, dash=(13, 2), width=4, fill='red')
        self.canvas.create_text(x1, y1, text=text, anchor=tk.SW, font=('Helvetica', 15), fill='red')

    def draw_img(self, x, y, img):
        new_img = ImageTk.PhotoImage(img)
        self.canvas.image = new_img
        self.canvas.create_image(x, y, anchor=tk.NW, image=new_img)


def create_button(master, text, command=None):
    config = {'text': text, 'font': ("Times", 20, 'bold')}

    if command:
        config['command'] = command
    button = tk.Button(master, config)

    return button


class Main(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root

        self.photos = []
        self.folder = None
        self.folder_button = None
        self.photos_button = None

        self.create_widgets()

    def set_folder(self):
        folder = askdirectory(parent=self.master, initialdir=os.getcwd(), title='Please select a folder with images')
        folder = os.path.abspath(folder)
        photos = tuple(folder + '\\' + i for i in os.listdir(folder))
        self.set_photos(photos)

    def set_photos(self, photos=None):
        if photos is None:
            photos = askopenfilenames(parent=self.master, initialdir=os.getcwd(), title='Please select a images')

        photos = tuple(
            os.path.abspath(i) for i in photos if re.search(r'\.(png|PNG|JPG|jpg|jpeg|JPEG)$', i) is not None)
        if photos:
            self.open_dialog(photos)

    def create_widgets(self):
        self.folder_button = create_button(self, 'Open Folder', self.set_folder)
        self.folder_button.pack(anchor=tk.W, padx=10, pady=10, side=tk.LEFT)

        self.photos_button = create_button(self, 'Browse Photos', self.set_photos)
        self.photos_button.pack(anchor=tk.NE, padx=10, pady=10, side=tk.RIGHT)

    def open_dialog(self, photos):
        Child(self, photos)


class Child(tk.Toplevel):
    def __init__(self, master, photos):
        super().__init__(master)

        self.width = 500
        self.height = 600
        self.canvas_size = (500, 500)
        self.GUI = GUI(self, *self.canvas_size)
        self.title('Viewer')
        self.geometry(f'{self.width}x{self.height}+100+100')
        self.resizable(False, False)

        self.grab_set()
        self.focus_set()

        self.folder_button = create_button(self, ' <  PREV', self.prev)
        self.folder_button.pack(anchor=tk.SW, padx=50, pady=30, side=tk.LEFT)

        self.photos_button = create_button(self, 'NEXT  > ', self.next)
        self.photos_button.pack(anchor=tk.SE, padx=50, pady=30, side=tk.RIGHT)

        self.photos = photos
        self.position = 0
        self.re_draw(self.position)

    def next(self):
        self.position += 1
        self.position %= len(self.photos)
        self.re_draw(self.position)

    def prev(self):
        self.position -= 1
        self.position %= len(self.photos)
        self.re_draw(self.position)

    def re_draw(self, position):
        self.GUI.canvas.delete("all")

        load = Image.open(self.photos[position])
        x, y, right, down = self.canvas_size + (-2, 0)
        width, height = load.size
        k = y / height
        width, height = k * width, y
        if width > x:
            k *= x / width
            width, height = x, x * height / width

        load = load.resize(size=(int(width), int(height)))

        dx, dy = (x - width) / 2 + right, (y - height) / 2 + down

        self.GUI.draw_img(int(dx), int(dy), load)

        for x1, y1, x2, y2, prediction in get_faces(self.photos[position]):
            x1, y1 = int(k * x1 + dx), int(k * y1 + dy)
            x2, y2 = int(k * x2 + dx), int(k * y2 + dy)
            self.GUI.draw(x1, y1, x2, y2, prediction)


def get_faces(path):
    image = cv2.imread(path)

    faces = detector.detect_faces(image)

    face_locations = tuple((x, y, x + width, y + height) for x, y, width, height in tuple(map(lambda _: _['box'], faces)))

    faces = tuple(image[top:bottom, left:right] for left, top, right, bottom in face_locations)

    result = tuple((*tuple(face_locations)[i], get_prediction(face)) for i, face in enumerate(faces))

    return result


def get_prediction(face):
    emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise')

    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = np.reshape(face, [1, face.shape[0], face.shape[1], 1])

    prediction_values = model.predict(face)

    prediction = int(np.argmax(prediction_values))

    predicted_label = emotions[prediction]

    return predicted_label


def main():
    root = tk.Tk()
    app = Main(root)
    app.pack()

    root.title('Face recognition')
    root.geometry('400x50+200+200')
    root.resizable(False, False)
    root.mainloop()


if __name__ == '__main__':
    main()
