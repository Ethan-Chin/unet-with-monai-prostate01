# To add a new cell, type '#'
# To add a new markdown cell, type '# [markdown]'
#
# from IPython import get_ipython

# [markdown]
# # 3D Segmentation with UNet
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Project-MONAI/tutorials/blob/master/3d_segmentation/unet_segmentation_3d_ignite.ipynb)
# [markdown]
# ## Setup environment

#
# from google.colab import drive
# drive.mount('/content/drive')


#
# get_ipython().system('python -c "import monai" || pip install -q "monai-weekly[ignite, nibabel, tensorboard]"')

# [markdown]
# ## Setup imports

#
# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
import glob
import logging
import os
# import shutil
import sys
# import tempfile

# import nibabel as nib
# import numpy as np
# from monai.config import print_config
from monai.data import (
    ArrayDataset, 
    # create_test_image_3d, 
    decollate_batch,
    )
from monai.handlers import (
    MeanDice,
    StatsHandler,
    stopping_fn_from_metric,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
)
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    # AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    CenterSpatialCrop,
    # RandSpatialCrop,
    # Resize,
    ScaleIntensity,
    EnsureType,
    Transpose,
)
# from monai.utils import first

import ignite
import torch


def unit_model(expert_id:int):

    #print_config()
    print(f"**********Now we are training for the expert {expert_id:02d}**********")
    print()
    # [markdown]
    # ## Setup data directory
    # 
    # You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.  
    # This allows you to save results and reuse downloads.  
    # If not specified a temporary directory will be used.

    #
    # directory = os.environ.get("/content/drive/MyDrive/unet-with-monai/dataset/prostate_gz")
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    directory = "/content/drive/MyDrive/unet-with-monai/dataset/prostate"
    root_dir = "/content/drive/MyDrive/unet-with-monai/dataset/prostate"
    print(directory)
    print()

    # [markdown]
    # ## Setup logging

    #
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # [markdown]
    # ## Setup data
    # First generate the proper file tree

    #
    training_dir = os.path.join(root_dir, 'Training')
    validation_dir = os.path.join(root_dir, 'Validation')
    # print(os.listdir(training_dir)[:10])
    # print(os.listdir(validation_dir)[:10])


    #
    # for i in range(40):
    #     im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)

    #     n = nib.Nifti1Image(im, np.eye(4))
    #     nib.save(n, os.path.join(root_dir, f"im{i}.nii.gz"))

    #     n = nib.Nifti1Image(seg, np.eye(4))
    #     nib.save(n, os.path.join(root_dir, f"seg{i}.nii.gz"))


    images_train = sorted(glob.glob(os.path.join(training_dir, "case*_image.nii.gz")))
    segs_train = sorted(glob.glob(os.path.join(training_dir, f"case*_task01_seg{expert_id:02d}.nii.gz")))

    images_val = sorted(glob.glob(os.path.join(validation_dir, "case*_image.nii.gz")))
    # segs_val = [sorted(glob.glob(os.path.join(validation_dir, f"case*_task01_seg{i:02d}.nii.gz"))) for i in range(6)]
    segs_val = sorted(glob.glob(os.path.join(validation_dir, f"case*_task01_seg{expert_id:02d}.nii.gz")))
    print("Dataset seted! The first 5 training cases and 3 validation cases:")
    for im, seg in zip(images_train[:5], segs_train[:5]):
        print(im, seg)
    print("--------------------")
    for im, seg in zip(images_val[:3], segs_val[:3]):
        print(im, seg)
    print("--------------------")
    print()

    # images = sorted(glob.glob(os.path.join(root_dir, "im*.nii.gz")))
    # segs = sorted(glob.glob(os.path.join(root_dir, "seg*.nii.gz")))

    # [markdown]
    # ## Setup transforms, dataset

    #
    # Define transforms for image and segmentation
    imtrans_train = Compose(
        [
            LoadImage(image_only=True),
            ScaleIntensity(),
            Transpose((2, 0, 1)),
            # AddChannel(),
            CenterSpatialCrop((640, 640)),
            # RandSpatialCrop((96, 96), random_size=False),
            EnsureType(),
        ]
    )
    segtrans_train = Compose(
        [
            LoadImage(image_only=True),
            Transpose((2, 0, 1)),
            # AddChannel(),
            CenterSpatialCrop((640, 640)),
            # RandSpatialCrop((96, 96), random_size=False),
            EnsureType(),
        ]
    )

    # Define nifti dataset, dataloader
    ds_train = ArrayDataset(images_train, imtrans_train, segs_train, segtrans_train)
    loader_train = torch.utils.data.DataLoader(
        ds_train, batch_size=4, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available()
    )
    # im, seg = first(loader_train)
    # print(im.shape, seg.shape)
    print("Dataset for training is loaded! the shape we may see:")
    for im, seg in loader_train:
        print(im.shape, seg.shape)
        break
    print()

    # [markdown]
    # ## Create Model, Loss, Optimizer

    #
    # Create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss = DiceLoss(sigmoid=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)

    # [markdown]
    # ## Create supervised_trainer using ignite

    #
    # Create trainer
    trainer = ignite.engine.create_supervised_trainer(
        net, opt, loss, device, False
    )
    print("Trainer is ready!")

    # [markdown]
    # ## Setup event handlers for checkpointing and logging

    #
    # optional section for checkpoint and tensorboard logging
    # adding checkpoint handler to save models (network
    # params and optimizer stats) during training

    log_dir = os.path.join(root_dir, f"logs_{expert_id:02d}")
    print('log_dir is set to: ', log_dir)
    print()

    # checkpoint_handler = ignite.handlers.ModelCheckpoint(
    #     log_dir, "net", n_saved=2, require_empty=False
    # )
    # trainer.add_event_handler(
    #     event_name=ignite.engine.Events.EPOCH_COMPLETED,
    #     handler=checkpoint_handler,
    #     to_save={"net": net, "opt": opt},
    # )

    # StatsHandler prints loss at every iteration
    # and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print
    # loss, user can also customize print functions
    # and can use output_transform to convert
    # engine.state.output if it's not a loss value
    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration
    # and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)

    # [markdown]
    # ## Add Validation every N epochs

    #
    # optional section for model validation during training
    validation_every_n_epochs = 1
    # Set parameters for validation
    metric_name = "Mean_Dice"
    # add evaluation metric to the evaluator engine
    val_metrics = {metric_name: MeanDice()}
    post_pred = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
    )
    post_label = Compose([EnsureType(), AsDiscrete(threshold_values=True)])
    # Ignite evaluator expects batch=(img, seg) and
    # returns output=(y_pred, y) at every iteration,
    # user can add output_transform to return other values
    evaluator = ignite.engine.create_supervised_evaluator(
        net,
        val_metrics,
        device,
        True,
        output_transform=lambda x, y, y_pred: (
            [post_pred(i) for i in decollate_batch(y_pred)],
            [post_label(i) for i in decollate_batch(y)]
        ),
    )

    # create a validation data loader
    imtrans_val = Compose(
        [
            LoadImage(image_only=True),
            ScaleIntensity(),
            Transpose((2, 0, 1)),
            #AddChannel(),
            #Resize((640, 640)),
            CenterSpatialCrop((640, 640)),
            EnsureType(),
        ]
    )
    segtrans_val = Compose(
        [
            LoadImage(image_only=True),
            Transpose((2, 0, 1)),
            #AddChannel(),
            #Resize((640, 640)),
            CenterSpatialCrop((640, 640)),
            EnsureType(),
        ]
    )



    ds_val = ArrayDataset(images_val, imtrans_val, segs_val, segtrans_val)
    loader_val = torch.utils.data.DataLoader(
        ds_val, batch_size=len(images_val), num_workers=2, pin_memory=torch.cuda.is_available()
    )

    print("Dataset for validation is loaded! the first shape we may see:")
    for im, seg in loader_val:
        print(im.shape, seg.shape)
        break
    print()


    @trainer.on(
        ignite.engine.Events.EPOCH_COMPLETED(every=validation_every_n_epochs)
    )
    def run_validation(engine):
        evaluator.run(loader_val)


    ###############################################################
    early_stopper = ignite.handlers.EarlyStopping(
        patience=20, score_function=stopping_fn_from_metric(metric_name), trainer=trainer, cumulative_delta=True
    )
    evaluator.add_event_handler(
        event_name= ignite.engine.Events.EPOCH_COMPLETED, handler=early_stopper
    )

    checkpoint_handler = ignite.handlers.ModelCheckpoint(
        log_dir, "net", n_saved=1, require_empty=False, score_function=stopping_fn_from_metric(metric_name), score_name=metric_name
    )
    evaluator.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        handler=checkpoint_handler,
        to_save={"net": net, "opt": opt},
    )
    ###############################################################

    # Add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        # no need to print loss value, so disable per iteration output
        output_transform=lambda x: None,
        # fetch global epoch number from trainer
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_stats_handler.attach(evaluator)

    # add handler to record metrics to TensorBoard at every validation epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        log_dir=log_dir,
        # no need to plot loss value, so disable per iteration output
        output_transform=lambda x: None,
        # fetch global epoch number from trainer
        global_epoch_transform=lambda x: trainer.state.epoch,
    )
    val_tensorboard_stats_handler.attach(evaluator)

    # add handler to draw the first image and the corresponding
    # label and model output in the last batch
    # here we draw the 3D output as GIF format along Depth
    # axis, at every validation epoch
    val_tensorboard_image_handler = TensorBoardImageHandler(
        log_dir=log_dir,
        batch_transform=lambda batch: (batch[0], batch[1]),
        output_transform=lambda output: output[0],
        global_iter_transform=lambda x: trainer.state.epoch,
    )
    evaluator.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED,
        handler=val_tensorboard_image_handler,
    )
    print("Evaluator is ready!")
    print("Now we start to train:")
    print("**"*10)

    # [markdown]
    # ## Run training loop

    #
    # create a training data loader

    max_epochs = 120
    state = trainer.run(loader_train, max_epochs)
    print("*"*10, "FINISHED!", "*"*10)
    print()
    print()

    # [markdown]
    # ## Visualizing Tensorboard logs

    #
    # get_ipython().run_line_magic('load_ext', 'tensorboard')
    # get_ipython().run_line_magic('tensorboard', '--logdir "/content/drive/MyDrive/unet-with-monai/dataset/prostate/logs"')

    # [markdown]
    # Expected training curve on TensorBoard:

    # [markdown]
    # ## Cleanup data directory
    # 
    # Remove directory if a temporary was used.

    #
    # if directory is None:
    #     shutil.rmtree(root_dir)
if __name__ == "__main__":
    unit_model(1)
