import icemapper
import tensorflow as tf
import dataloaders
import dataloaders.transformations
import h5py
import utils


def main():
    features = ["grd", "insar"]
    model_name = "20241009_ICEmapper_v2_grdinsar"
    
    data_folder = "/extravolume/data/icemapper"
    timesteps = 15
    patch_size = 384
    batch_size = 8
    
    learning_rate = 5e-4
    
    # set up dataloaders
    train_dataloader, steps_per_epoch = dataloaders.DataLoader(
        dataset_path=f"{data_folder}/train.hdf5", 
        mode="train",
        patch_size=patch_size, 
        features=features,
        batch_size=batch_size, 
        prefetch=1,
    )
    
    val_dataloader, _ = dataloaders.DataLoader(
        dataset_path=f"{data_folder}/val.hdf5", 
        mode="val",
        patch_size=patch_size, 
        features=features,
        batch_size=batch_size, 
        prefetch=1,
    )
    
    # set up model
    model = icemapper.ICEmapper_v2(
        (timesteps, patch_size, patch_size, len(features)), 2,
        dropout=0.05,
        name=model_name,
    )
    model.summary()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=utils.FocalLoss(),
        # loss=utils.DistancePenaltyFocalLoss(),
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
        # tf.keras.callbacks.ModelCheckpoint(
        #     f"weights/{model_name}_minloss.h5",
        #     monitor=f"val_loss",
        #     mode="min",
        #     save_best_only=True,
        #     save_weights_only=True,
        # ),
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
    
