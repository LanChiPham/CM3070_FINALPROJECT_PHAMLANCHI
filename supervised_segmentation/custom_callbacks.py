import os
import cv2
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class GifCreator:
    def __init__(self, test_file, output_dir='gifs', fps=15):
        self.stop_training = False
        self.test_file = test_file
        self.image = cv2.imread(self.test_file, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, (256, 256))
        self.image = (self.image - 127.50)/127.50
        # image_mean = np.mean(self.image)
        # image_std = np.std(self.image)
        # self.image = (self.image - image_mean) / image_std

        if len(self.image.shape) < 3:
            img = self.image
            self.image = np.zeros((self.image.shape[0], self.image.shape[1], 3))
            self.image[:, :, 0] = img
            self.image[:, :, 1] = img
            self.image[:, :, 2] = img

        # Normalize the image to the [0, 1] range
        # max_pixel_value = 255  # Assuming 8-bit images
        # self.image = self.image / max_pixel_value
        
        self.output_dir = output_dir
        self.fps = fps
        self.counter = 0

        if not os.path.exists(self.output_dir):
            print('Creating GIF output directory ', self.output_dir)
            os.mkdir(self.output_dir)
        else:
            # Clear the output directory
            for _file in glob.glob(f'{self.output_dir}/*.png'):
                os.remove(_file)

            print('Gif output directory cleared')

    def log_output(self, model):
        if(self.model is not None):
            self.counter += 1
            output, logits = self.model.predict(np.array([self.image]))
            output = output[0]
            output = output * 255.0
            output = output.astype(np.uint8)

            output_file = f'{self.output_dir}/{self.counter}.png'
            fig, ax = plt.subplots(1, 2, figsize=(15, 8))
            ax[0].imshow(self.image)
            ax[0].set_title('Input Image')
        
            ax[1].imshow(output)
            ax[1].set_title('Segmentation map')
            plt.savefig(output_file)

    def reset_state(self, state):
        self.model = state['model']

    def __call__(self):
        self.log_output(self.model)

class EarlyStopping:
    def __init__(self, patience, monitor='mean_val_loss'):
        self.stop_training = False
        self.patience = patience
        self.monitor = monitor
        self.losses = []

    @staticmethod
    def _overfitting(losses, patience):
        overfitted = False
        if(len(losses) >= patience):
            head = losses[-patience:]
            overfitted = sorted(head) == head

        return overfitted

    def reset_state(self, state):
        if(self.monitor not in state):
            raise Exception(f'"{self.monitor}" monitor is not in state dict ...')

        self.losses.append(state[self.monitor])
        
    def __call__(self):
        if(EarlyStopping._overfitting(self.losses, self.patience)):
            self.stop_training = True

class InfoLogger:
    def __init__(self, log_dir, log_name):
        self.stop_training = False
        self.log_dir = log_dir
        self.log_name = log_name

        if(not os.path.exists(self.log_dir)):
            os.mkdir(self.log_dir)

        if(not os.path.exists(f'{self.log_dir}/{self.log_name}')):
            os.mkdir(f'{self.log_dir}/{self.log_name}')

        self.save_path = os.path.join(self.log_dir, self.log_name)
        
        ### Reset log if exists ###
        for _file in glob.glob(f'{self.save_path}/*.csv'):
            os.unlink(_file)

        self.mean_train_losses = []
        self.mean_val_losses = []

    def reset_state(self, state):
        self.mean_train_losses.append(state['mean_train_loss'])
        self.mean_val_losses.append(state['mean_val_loss'])

    def __call__(self):
        print('Logging training info ...')
        loss_df = pd.DataFrame({'train' : self.mean_train_losses, 'val' : self.mean_val_losses})
        loss_df.to_csv(f'{self.save_path}/losses.csv')