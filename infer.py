import math
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model.model_1 import Net

save_path = "./pt-torch.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("deivce use", device)

model = Net().to(device)
model.load_state_dict(torch.load(save_path))
model.eval()

def preprocess(image: Image):

    # Convert the image to grayscale
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    threshold_value = 200
    binary_array = 255 - np.where(image_array > threshold_value, image_array, 0)

    # to PIL again (maybe dont need to do this)
    binary_image = Image.fromarray(binary_array.astype('uint8'))
    # show it
    # print(binary_array)
    # binary_image.show()
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    
    return transform(binary_image).unsqueeze(0)


def predict_digit(image):
    # Preprocess the image
    preprocessed_image = preprocess(image)
    
    # Use the model to predict the digit
    with torch.no_grad():
        output = model(preprocessed_image.to(device))
        pred = output.argmax(dim=1, keepdim=True)
    
    # Return the predicted digit
    return pred.item()

class HandDraw:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.canvas = self.fig.canvas
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        self.drawing = False
        self.last_pos = None
        self.current_pos = None
        self.lock = False
        self.reset()

    def reset(self):
        self.ax.clear()
        self.ax.set_xlim([0, 1])
        self.ax.set_ylim([0, 1])
        plt.axis('off')

    def press(self, event):
        if event.key == '0' or event.key == 'r':
            if self.lock:
                return
            self.lock = True
            self.reset()
            self.canvas.draw()
            self.lock = False

    def on_press(self, event):
        self.drawing = True
        self.current_pos = (event.xdata, event.ydata)

    def on_motion(self, event):
        if self.drawing:
            self.last_pos = self.current_pos
            self.current_pos = (event.xdata, event.ydata)
            if self.last_pos is not None:
                self.ax.plot(
                    [self.last_pos[0], self.current_pos[0]],
                    [self.last_pos[1], self.current_pos[1]],
                    color='black', linewidth=8
                )
            self.canvas.draw()

    def on_release(self, event):
        self.drawing = False
        self.last_pos = None
        self.current_pos = None
        if self.lock:
            return
        self.lock = True
        self.predict()
        self.lock = False
    

    def predict(self):

        # print(self.fig.canvas.tostring_rgb())
        buf = self.fig.canvas.tostring_rgb()

        # maybe enlarge 1.25 / 1.5 / ... times
        all_len = len(buf)
        ncols, nrows = self.fig.canvas.get_width_height()
        enlarge = math.sqrt(all_len / (ncols * nrows * 3))

        image = Image.frombytes('RGB', (int(ncols * enlarge), int(nrows * enlarge)), self.fig.canvas.tostring_rgb())
        digit = predict_digit(image)

        # Print the predicted digit
        print("Predicted digit:", digit)


hd = HandDraw()
plt.show()
