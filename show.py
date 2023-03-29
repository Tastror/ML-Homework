import os
import torch
import argparse


parser = argparse.ArgumentParser(description='show the model')
parser.add_argument(
    'model_name', metavar='model', type=str, nargs='?',
    default="pt-torch-last.pt",
    help='model name, save dir is weight/, default is pt-torch-last.pt'
)
args = parser.parse_args()
print('show weight/', args.model_name, sep="")
save_dir = 'weight'
last_save_path = os.path.join(save_dir, args.model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(last_save_path)
print(model.to(device))
