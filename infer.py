import os
import math
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='use the model to inference')
parser.add_argument(
    'model_name', metavar='model', type=str, nargs='?',
    default="pt-torch-best.pt",
    help='model name, save dir is weight/ (default: pt-torch-best.pt)'
)
args = parser.parse_args()
print('use weight/', args.model_name, sep="")


data_dir = 'dataset'
data_name = 'NewDataset.txt'
data_path = os.path.join(data_dir, data_name)
data_description = []
data_file = open(data_path)
line = True
while line:
    line = data_file.readline()
    data_description.append(line.strip())
data_file.close()


save_dir = 'weight'
last_save_path = os.path.join(save_dir, args.model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("deivce use", device)
model = torch.load(last_save_path)
model.eval()


def preprocess(image: Image) -> torch.Tensor:

    # 反色归一预处理
    image = image.convert('L')
    image = image.resize((28, 28))
    image = 1 - np.array(image, dtype=np.float32) / 255

    # 画板颜色比较淡，要加强一点
    image = np.sqrt(image)
    image = np.where(image > 0.5, 1, image)

    # 查看效果
    # Image.fromarray(np.uint8(image * 255)).show()

    # 扩展为张量
    image = np.expand_dims(image, axis=2)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0)

    return image


def predict_digit(image):

    preprocessed_image = preprocess(image)

    with torch.no_grad():
        output = model(preprocessed_image.to(device))
        pred = output.argmax(dim=1, keepdim=True)

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

        buf = self.fig.canvas.tostring_rgb()

        # maybe enlarge 1.25 / 1.5 / ... times
        all_len = len(buf)
        ncols, nrows = self.fig.canvas.get_width_height()
        enlarge = math.sqrt(all_len / (ncols * nrows * 3))

        image = Image.frombytes(
            'RGB', (int(ncols * enlarge), int(nrows * enlarge)),
            self.fig.canvas.tostring_rgb()
        )
        digit = predict_digit(image)

        # Print the predicted digit
        print("Predicted:", digit)
        print(data_description[digit])
        print()


hd = HandDraw()
plt.show()
