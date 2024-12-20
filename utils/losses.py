import tensorflow as tf
import scipy.ndimage
import numpy as np


class FocalLoss(tf.keras.losses.Loss):
    def __init__(
        self, gamma=2.0, alphas=None, reduction=tf.keras.losses.Reduction.AUTO, name="focal_loss", **kwargs
    ):
        super(FocalLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.gamma = gamma
        self.alphas = alphas

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1 - epsilon)
        ce = y_true * tf.math.log(y_pred)
        loss = -tf.math.pow(1 - y_pred, self.gamma) * ce
        if self.alphas is not None:
            loss = self.alphas * loss
        focal_loss = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=-1))
        return focal_loss


class DistancePenaltyFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self, gamma=2.0, alphas=None, penalty_weight=1.0, reduction=tf.keras.losses.Reduction.AUTO, name="distance_penalty_focal_loss", **kwargs
    ):
        super(DistancePenaltyFocalLoss, self).__init__(reduction=reduction, name=name, **kwargs)
        self.gamma = gamma
        self.alphas = alphas
        self.penalty_weight = penalty_weight
        
        def get_distance_map(y_true):
            INF = 99999
            batch_size, height, width, channels = y_true.shape
            distance_map = np.zeros((batch_size, height, width, 1))
            for item_idx in range(batch_size):
                for channel_idx in range(channels):
                    class_map = y_true[item_idx, :, :, channel_idx]
                    if (class_map == 1.0).all():
                        distance_map[item_idx, :, :, 0] += INF
                    else:
                        distance_map[item_idx, :, :, 0] += scipy.ndimage.distance_transform_edt(class_map)
            return distance_map
        self.get_distance_map = get_distance_map

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.keras.backend.clip(y_pred, epsilon, 1 - epsilon)
        ce = y_true * tf.math.log(y_pred)
        loss = -tf.math.pow(1 - y_pred, self.gamma) * ce
        if self.alphas is not None:
            loss = self.alphas * loss
        distance_map = self.get_distance_map(y_true)
        distance_weight = 1 / (1 + distance_map)
        loss += (self.penalty_weight * distance_weight * loss)
        loss = tf.math.reduce_mean(tf.math.reduce_sum(loss, axis=-1))
        return loss
