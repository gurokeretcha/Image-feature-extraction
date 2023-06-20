"""
generated image features
"""
import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

patch_size = 14 # patchsize=14
#520//14
patch_h  = 520//patch_size
patch_w  = 520//patch_size

# feat_dim = 384 # vits14
# feat_dim = 768 # vitb14
# feat_dim = 1024 # vitl14
feat_dim = 1536 # vitl14

#520//14
patch_h  = 520//patch_size
patch_w  = 520//patch_size

def list_files(dataset_path):
    images = []
    for root, _, files in os.walk(dataset_path):
        for name in files:
            images.append(os.path.join(root, name))
    return images


class CustomImageDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = list_files(self.img_dir)
        self.transform =  transforms.Compose([           
                                transforms.Resize(520),
                                transforms.CenterCrop(518), #should be multiple of model patch_size                 
                                transforms.ToTensor(),                    
                                transforms.Normalize(mean=0.5, std=0.2)
                                ])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


def extract_features(data_path, model_name):
    dataset = CustomImageDataset(data_path)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    print('total number of images: ', len(dataset))
    print('total number of batched: ', len(train_dataloader))
    try:
        if model_name not in ['dinov2_small','dinov2_medium','dinov2_large']:
            raise ValueError("Invalid input. Please choose from the available options.")

        if model_name=='dinov2_small':
            dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif model_name=='dinov2_medium':
            dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        elif model_name=='dinov2_large':
            dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        else:
            print('please insert correct name of the model. (available names: dinov2_small','dinov2_medium','dinov2_large')
        final_img_features = []
        final_img_filepaths = []
        total_features  = []
        with torch.no_grad():
            for image_tensors, file_paths in tqdm(train_dataloader):
                try:
                    features_dict = dinov2_vitl14.forward_features(image_tensors)
                    features = features_dict['x_norm_patchtokens']
                    total_features.append(features)

                    image_features = dinov2_vitl14(image_tensors) #384 small, #768 base, #1024 large
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_features = image_features.tolist()
                    final_img_features.extend(image_features)
                    final_img_filepaths.extend((list(file_paths)))

                except Exception as e:
                    print("Exception occurred: ", e)
                    break
    except Exception as e:
        print("An error occurred:", str(e))

    total_features = torch.cat(total_features, dim=0)
    return np.array(final_img_features), final_img_filepaths, total_features


def main():
    dir_path = "data/"
    extraction, filepaths,total_features = extract_features(dir_path, 'dinov2_small')
    print(filepaths)
    print(total_features.shape)
    
if __name__=="__main__":
    main()

