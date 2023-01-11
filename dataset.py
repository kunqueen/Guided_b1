import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image

class Ffhq(Dataset):
     def __init__(self, img_dir, transform):
          self.img_dir = img_dir
          self.transform = transform
          # self.mode = mode
          self.train_dataset = []
          self.test_dataset = []
          self.file_list = []


     def __len__(self):
          return len(self.train_dataset)
     
     def __getitem__(self, index):
          self.train_data_dir = self.img_dir
          self.file_list.extend(os.listdir(self.train_data_dir))

          for i, file in enumerate(self.file_list):
               self.train_dataset.append([file])
          dataset = self.train_dataset
          filename = dataset[index]
          
          image = Image.open(os.path.join(self.train_data_dir, filename))

          return self.transform(image)




# training_data = datasets.FashionMNIST(
#      root="data",
#      train=True,
#      download=True,
#      transform=ToTensor()
# )

# test_data = datasets.FashionMNIST(
#      root="data",
#      train=False,
#      download=True,
#      transform=ToTensor()
# )





# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# print(len(train_dataloader))

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label:{label}")


