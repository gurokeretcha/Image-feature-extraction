from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from image_feature_extractor.feature_extraction import extract_features 
import os 
from PIL import Image
patch_size = 14 # patchsize=14

# patch_h = 518//patch_size
# patch_w = 518//patch_size

# # feat_dim = 384 # vits14
# # feat_dim = 768 # vitb14
# feat_dim = 384 # vitl14
# # feat_dim

def feature_vis(features, folder_path):
    features = features[:4,:,:]
    patch_h = 518//patch_size
    patch_w = 518//patch_size
    feat_dim = features.shape[-1]
    img_num = features.shape[0]

    total_features = features.reshape(img_num * patch_h * patch_w, feat_dim) #4(*H*w, 1024)
    pca = PCA(n_components=3)
    pca.fit(total_features)
    pca_features = pca.transform(total_features)

    # min_max scale
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                        (pca_features[:, 0].max() - pca_features[:, 0].min())
    #pca_features = sklearn.processing.minmax_scale(pca_features)

        # segment/seperate the backgound and foreground using the first component
    pca_features_bg = pca_features[:, 0] > 0.45 # from first histogram
    pca_features_fg = ~pca_features_bg



        # 2nd PCA for only foreground patches
    pca.fit(total_features[pca_features_fg]) 
    pca_features_left = pca.transform(total_features[pca_features_fg])

    for i in range(3):
        # min_max scaling
        pca_features_left[:, i] = (pca_features_left[:, i] - pca_features_left[:, i].min()) / (pca_features_left[:, i].max() - pca_features_left[:, i].min())

    pca_features_rgb = pca_features.copy()
    # for black background
    pca_features_rgb[pca_features_bg] = 0
    # new scaled foreground features
    pca_features_rgb[pca_features_fg] = pca_features_left

    # reshaping to numpy image format
    pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)
    fig, axs = plt.subplots(4, 4,figsize=(10, 10))
    
    img_path = os.listdir(folder_path)

    print(img_path)
    for i in range(4):
        img = Image.open('data/'+img_path[i]).convert('RGB').resize((700, 700))
        axs[0, i].imshow(img)
        
        axs[1, i].imshow(pca_features[i*patch_h*patch_w : (i+1)*patch_h*patch_w, 0].reshape(patch_h, patch_w))

        axs[2, i].imshow(pca_features_bg[i * patch_h * patch_w: (i+1) * patch_h * patch_w].reshape(patch_h, patch_w))

        axs[3, i].imshow(pca_features_rgb[i])

    plt.tight_layout()
    plt.show()

    return pca_features
def main():
    data_path = 'data/'
    # total_features = extract_features(data_path, 'dinov2_small')
    # pca_f = PCA_func(total_features)
    # visualizee(pca_f)

if __name__=="__main__":
    main()
