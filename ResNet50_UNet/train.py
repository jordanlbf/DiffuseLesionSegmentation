from fastai.vision.all import *
from torch import nn
from PIL import Image
import numpy as np
from torchvision.transforms import ColorJitter
from fastai.losses import BaseLoss
from torch.nn import functional as F
from data_loader import get_y_fn, preprocess_mask, rename_to_uppercase_jpg, get_aug_pipeline
from show_training import show_training
from model import create_unet


def train(image_path, size_tform, lr, load, save, pat):
    path = Path('')
    codes = np.array(['skin', 'lesion'])
    fnames = get_image_files(path / image_path)
    if not fnames:
        raise ValueError(
            f"No images found. Check your dataset path and file extensions. Searched path: {path / image_path}")

    t_forms = get_aug_pipeline(size_tform)

    dls = SegmentationDataLoaders.from_label_func(
        path, bs=8, fnames=fnames, label_func=get_y_fn, codes=codes,
        item_tfms=Resize(size_tform),
        batch_tfms=t_forms,
        num_workers=0
    )

    learn = create_unet(dls, output_channels=2)

    if load is not None:
        learn.load(load)

    show_training(learn, size_tform)
    return

    # ReduceLROnPlateau Callback
    reduce_lr = ReduceLROnPlateau(monitor='valid_loss', factor=0, patience=1)

    # Training with EarlyStopping and ReduceLROnPlateau callbacks
    learn.fine_tune(200, base_lr=lr,
                    cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=pat),
                         reduce_lr,
                         SaveModelCallback(monitor='valid_loss', fname=save)])

    # Save the final model
    learn.save(save)



def main():
    path = os.path.join(os.path.dirname(os.getcwd()), '', 'data')
    # Example train call with additional parameters for ReduceLROnPlateau
    train(path, 1024, 0.0008, "ResNet34", "ResNet34", 10)
    # train('ISIC/ISIC/images', 256, 0.0008, None, "ISIC_Deeplab_ResNet101_260", 10)
    # train('Multiple/images', 420, 0.00008, "Multiple_Deeplab_ResNet_280", "Multiple_Deeplab_ResNet_280", 10)
    # train('Diffuse/images', 512, 0.00004, "Diffuse_Deeplab_ResNet_320_2", "Diffuse_Deeplab_ResNet_320_2", 10)


if __name__ == '__main__':
    main()