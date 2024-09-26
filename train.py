import icemapper
import tensorflow as tf
import dataloaders
import dataloaders.transformations
import h5py
import utils


def main():
    features = ["grd"]
    model_name = "20240925_ICEmapper_v1_grdonly"
    
    data_folder = "/extravolume/data/icemapper"
    timesteps = 15
    patch_size = 384
    batch_size = 8
    
    learning_rate = 5e-4
    
    # set up dataloaders
#     train_data = h5py.File(f"{data_folder}/train.hdf5", "r")
#     val_data = h5py.File(f"{data_folder}/val.hdf5", "r")
    
#     train_sampler = dataloaders.RandomSampler(train_data, patch_size, features)
#     train_sampler = dataloaders.move_sampler_to_ram(train_sampler, keys=features + ["outlines"])
#     train_dataloader, steps_per_epoch = dataloaders.DataLoader(
#         train_sampler,
#         plugins=[
#             dataloaders.ShuffleTimeIndex(p=0.2),
#             dataloaders.Augmentation([
#                 dataloaders.transformations.random_vertical_flip(p=0.5),
#                 dataloaders.transformations.random_horizontal_flip(p=0.5),
#                 dataloaders.transformations.random_rotation(p=0.75),
#                 dataloaders.transformations.crop_and_scale(patch_size=patch_size, scale=0.8, p=0.25),
#                 dataloaders.transformations.add_gamma_correction(gamma_l=0.8, gamma_r=1.2, p=0.25),
#                 dataloaders.transformations.add_gaussian_shift(sigma=0.025, p=0.25), 
#             ]),
#             dataloaders.AddLabelSmoothing(),
#         ],
#         batch_size=batch_size,
#     )
    
#     val_sampler = dataloaders.ConsecutiveSampler(val_data, patch_size, features)
#     val_sampler = dataloaders.move_sampler_to_ram(val_sampler, keys=features + ["outlines"])
#     val_dataloader, _ = dataloaders.DataLoader(
#         val_sampler,
#         plugins=[],
#         batch_size=batch_size,
#     )
    train_dataloader, steps_per_epoch = dataloaders.DataLoader(
        dataset_path=f"{data_folder}/train.hdf5", 
        mode="train",
        patch_size=patch_size, 
        features=features,
        batch_size=batch_size, 
    )
    
    val_dataloader, _ = dataloaders.DataLoader(
        dataset_path=f"{data_folder}/val.hdf5", 
        mode="val",
        patch_size=patch_size, 
        features=features,
        batch_size=batch_size, 
    )
    val_dataloader = val_dataloader.cache()
    
    # set up model
    model = icemapper.ICEmapper(
        (timesteps, patch_size, patch_size, len(features)), 2,
        pooling=tf.keras.layers.MaxPooling3D,
        dropout=0.1,
        name=model_name,
    )
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=utils.FocalLoss(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            utils.IoU(class_idx=1),
        ]
    )
    
    # fit model
    callbacks = [
        utils.LRRestartsWithCosineDecay(
            start=learning_rate, 
            restart_steps=[steps_per_epoch * _ for _ in [10, 30, 70, 150]],
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f"weights/{model_name}.h5",
            monitor=f"val_iou",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.CSVLogger(
            f"logs/{model_name}.csv",
        ),
    ]
    
    model.fit(
        train_dataloader,
        epochs=150,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataloader,
        callbacks=callbacks,
        verbose=1,
    )
    

if __name__ == "__main__":
    main()
    
