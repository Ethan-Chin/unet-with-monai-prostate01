import glob
import logging
import os

import sys

from monai.data import (
    ArrayDataset,
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
    AsDiscrete,
    Compose,
    LoadImage,
    CenterSpatialCrop,
    ScaleIntensity,
    EnsureType,
    Transpose,
)

import ignite
import torch


def unit_model(expert_id: int, param_device):

    print(f"**********Now we are training for the expert {expert_id:02d}**********")
    print()

    data_dir = "./dataset/working_data"
    if not os.path.exists("./exp_new"):
        os.mkdir("./exp_new")
    root_dir = "./exp_new"
    print()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    training_dir = os.path.join(data_dir, 'Training')
    validation_dir = os.path.join(data_dir, 'Validation')


    images_train = sorted(glob.glob(os.path.join(training_dir, "case*", "image.nii.gz")))
    segs_train = sorted(glob.glob(os.path.join(training_dir, "case*", f"task01_seg{expert_id:02d}.nii.gz")))

    images_val = sorted(glob.glob(os.path.join(validation_dir, "case*", "image.nii.gz")))
    segs_val = sorted(glob.glob(os.path.join(validation_dir, "case*", f"task01_seg{expert_id:02d}.nii.gz")))
    print("Dataset seted! The first 5 training cases and 3 validation cases:")
    for im, seg in zip(images_train[:5], segs_train[:5]):
        print(im, seg)
    print("--------------------")
    for im, seg in zip(images_val[:3], segs_val[:3]):
        print(im, seg)
    print("--------------------")
    input("Please Check if it is correct: ")

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


    ds_train = ArrayDataset(images_train, imtrans_train, segs_train, segtrans_train)
    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

    print("Dataset for training is loaded! the shape we may see:")
    for im, seg in loader_train:
        print(im.shape, seg.shape)
        break
    print()

    device = param_device
    net = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss = DiceLoss(include_background=False, sigmoid=True)
    lr = 1e-3
    opt = torch.optim.Adam(net.parameters(), lr)


    trainer = ignite.engine.create_supervised_trainer(
        net, opt, loss, device, False
    )
    print("Trainer is ready!")


    log_dir = os.path.join(root_dir, f"logs_{expert_id:02d}")
    print('log_dir is set to: ', log_dir)
    print()


    train_stats_handler = StatsHandler(name="trainer", output_transform=lambda x: x)
    train_stats_handler.attach(trainer)

    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir=log_dir, output_transform=lambda x: x)
    train_tensorboard_stats_handler.attach(trainer)


    validation_every_n_epochs = 1

    metric_name = "Mean_Dice"

    val_metrics = {metric_name: MeanDice(include_background=False)}
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
            # AddChannel(),
            # Resize((640, 640)),
            CenterSpatialCrop((640, 640)),
            EnsureType(),
        ]
    )
    segtrans_val = Compose(
        [
            LoadImage(image_only=True),
            Transpose((2, 0, 1)),
            # AddChannel(),
            # Resize((640, 640)),
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
        patience=40, score_function=stopping_fn_from_metric(metric_name), trainer=trainer, cumulative_delta=True
    )
    evaluator.add_event_handler(
        event_name=ignite.engine.Events.EPOCH_COMPLETED, handler=early_stopper
    )

    checkpoint_handler = ignite.handlers.ModelCheckpoint(
        log_dir, "net", n_saved=1, require_empty=False, score_function=stopping_fn_from_metric(metric_name),
        score_name=metric_name
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

    print("Evaluator is ready!")
    print("Now we start to train:")
    print("**" * 10)


    max_epochs = 150
    state = trainer.run(loader_train, max_epochs)
    print("*" * 10, "FINISHED!", "*" * 10)
    print()
    print()


if __name__ == "__main__":
    os.chdir("../")
    exp_id = int(sys.argv[1])
    device = torch.device(sys.argv[2])
    unit_model(exp_id, device)
    # example: $ python ./unet_unit.py 2 cuda:0
    # means train the 2ed (expert 2) unet model on cuda 0
