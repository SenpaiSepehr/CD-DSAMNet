import numpy as np
from PIL import Image

import os
from train_options import parser
opt = parser.parse_args()

class BinaryImageConverter:
    def __init__(self, tensor):
        self.tensor = tensor

    def convert_and_save_images(self, save_path):
        # Reshape the tensor to (batch_size, width, height)
        batch_size, _, width, height = self.tensor.shape
        images = self.tensor.reshape(batch_size, width, height)

        for i in range(batch_size):

            image = images[i]
            image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='L')

            image_path = os.path.join(save_path, f'image_{i}.png')

            # Save the image as PNG
            image.save(image_path)



class BinaryChannelConverter:
    def __init__(self, predict):
        self.predict = predict

    def convert_and_save_images(self, save_path):
        # Reshape the tensor to (batch_size, width, height)
        batch_size, channels, width, height = self.predict.shape
        #images = self.predict.reshape(batch_size,channels,width, height)
        images = self.predict
        
        for batch in range(batch_size):
            for channel in range(channels):
                
                image = images[batch,channel]
                image = (image.detach().cpu().numpy() * 255).astype(np.uint8)
                image = Image.fromarray(image, mode='L')
                
                image_path = os.path.join(save_path, f'channel_img_{channel}.png')
                
                # Save the image as PNG
                image.save(image_path)
