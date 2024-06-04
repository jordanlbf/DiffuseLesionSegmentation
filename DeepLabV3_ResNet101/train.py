from fastai.vision.all import *
from pathlib import Path
import numpy as np
from data_loader import get_y_fn, get_aug_pipeline
from model import create_deeplab
from show_training import show_training
from model_v3p import create_deeplabv3plus

def train(image_path, size_tform, lr, load, save, pat):
    path = Path('')
    codes = np.array(['skin', 'lesion'])
    fnames = get_image_files(path / image_path)
    if not fnames:
        raise ValueError("No images found. Check your dataset path and file extensions.")

    t_forms = get_aug_pipeline(size_tform)

    dls = SegmentationDataLoaders.from_label_func(
        path, bs=8, fnames=fnames, label_func=get_y_fn, codes=codes,
        item_tfms=Resize(size_tform),
        batch_tfms=t_forms,
        num_workers=0
    )

    # deeplab = create_deeplabv3plus(output_channels=2)

    """This is needed for the DeepLabV3 Model"""
    deeplab = create_deeplab(output_channels=2)

    # Initialize the Learner without ProgressCallback
    learn = Learner(dls, deeplab, loss_func=CrossEntropyLossFlat(axis=1), metrics=[DiceMulti()],
                    opt_func=Adam, cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=pat))

    if load is not None:
        learn.load(load)
    show_training(learn, size_tform)
    return

    learn.fine_tune(200, base_lr=lr)

    learn.save(save)
    show_training(learn, size_tform)



def main():
    """Change these variables accordingly"""
    path = os.path.join(os.path.dirname(os.getcwd()), '', 'data')
    #train('ISIC/ISIC/images', 256, 0.0008, None, "ISIC_Deeplab_ResNet101_260", 10)
    train(path, 1024, 0.001, "DeepLabv3_ResNet101_Final", "DeepLabv3_ResNet101_Final2", 10)
    #train('Multiple/images', 420, 0.00008, "Multiple_Deeplab_ResNet_280", "Multiple_Deeplab_ResNet_280", 10)
    #train('Diffuse/images', 512, 0.00004, "Diffuse_Deeplab_ResNet_320_2", "Diffuse_Deeplab_ResNet_320_2", 10)

if __name__ == '__main__':
    main()
