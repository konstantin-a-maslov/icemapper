import tensorflow as tf


class Residual(tf.keras.layers.Layer):
    def __init__(self, layer):
        super(Residual, self).__init__()
        self.layer = layer

    def call(self, x):
        x1 = self.layer(x)
        y = tf.math.add(x, x1)
        return y


class ResidualWithProjection(tf.keras.layers.Layer):
    def __init__(self, layer, projected_size):
        super(ResidualWithProjection, self).__init__()
        self.layer = layer
        self.projected_size = projected_size
        self.projection = tf.keras.layers.Dense(projected_size, use_bias=False)

    def call(self, x):
        x1 = self.layer(x)
        x = self.projection(x)
        y = tf.math.add(x, x1)
        return y


class ConvBatchNormAct_x2(tf.keras.layers.Layer):
    def __init__(
        self, n_filters, kernel_size=3, dilation_rate=1, use_bias=True, padding="same",
        activation=tf.keras.layers.LeakyReLU, spatial_dropout=0, mcdropout=False
    ):
        super(ConvBatchNormAct_x2, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            n_filters,
            kernel_size,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            padding=padding
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = activation()
        self.conv2 = tf.keras.layers.Conv2D(
            n_filters,
            kernel_size,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            padding=padding
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = activation()
        self.spatial_dropout = spatial_dropout
        self.mcdropout = mcdropout
        if spatial_dropout:
            self.dropout = tf.keras.layers.SpatialDropout2D(spatial_dropout)

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        if self.spatial_dropout:
            y = self.dropout(y, training=self.mcdropout)
        return y


class UpConv(tf.keras.layers.Layer):
    def __init__(
        self, n_filters, kernel_size=2, dilation_rate=1, use_bias=False, padding="same"
    ):
        super(UpConv, self).__init__()
        self.upsampling = tf.keras.layers.UpSampling2D(
            size=(kernel_size, kernel_size)
        )
        self.conv = tf.keras.layers.Conv2D(
            n_filters,
            kernel_size,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            padding=padding
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        y = self.upsampling(x)
        y = self.conv(y)
        y = self.bn(y)
        return y


class UpConv3D(tf.keras.layers.Layer):
    def __init__(
        self, n_filters, kernel_size=2, dilation_rate=1, use_bias=False, padding="same"
    ):
        super(UpConv3D, self).__init__()
        self.upsampling = tf.keras.layers.UpSampling3D(
            size=(kernel_size, kernel_size, kernel_size)
        )
        self.conv = tf.keras.layers.Conv3D(
            n_filters,
            kernel_size,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            padding=padding
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        y = self.upsampling(x)
        y = self.conv(y)
        y = self.bn(y)
        return y


class Conv3DBatchNormAct_x2(tf.keras.layers.Layer):
    def __init__(
        self, n_filters, kernel_size=3, dilation_rate=1, use_bias=True, padding="same",
        activation=tf.keras.layers.LeakyReLU, spatial_dropout=0, mcdropout=False
    ):
        super(Conv3DBatchNormAct_x2, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(
            n_filters,
            kernel_size,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            padding=padding
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = activation()
        self.conv2 = tf.keras.layers.Conv3D(
            n_filters,
            kernel_size,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            padding=padding
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = activation()
        self.spatial_dropout = spatial_dropout
        self.mcdropout = mcdropout
        if spatial_dropout:
            self.dropout = tf.keras.layers.SpatialDropout3D(spatial_dropout)

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        if self.spatial_dropout:
            y = self.dropout(y, training=self.mcdropout)
        return y
    

class TemporalWeightedPooling(tf.keras.layers.Layer):
    def __init__(
        self, pool_size, activation=tf.keras.activations.swish, padding="valid"
    ):
        super(TemporalWeightedPooling, self).__init__()
        self.pool_size = pool_size
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size, pool_size)
        self.activation = activation
        self.padding = padding
        
    def build(self, input_shape):
        n_filters = input_shape[-1]
        
        self.spatial_pooling = tf.keras.layers.MaxPooling3D(
            pool_size=(1, self.pool_size[1], self.pool_size[2]), padding=self.padding
        )
        self.global_pooling = tf.keras.layers.TimeDistributed(
            tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        )
        self.conv1 = tf.keras.layers.Conv3D(
            n_filters // 2, (3, 1, 1), use_bias=True, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv3D(
            n_filters, (3, 1, 1), use_bias=True, padding="same"
        )
        self.temporal_pooling = tf.keras.layers.AveragePooling3D(
            pool_size=(self.pool_size[0], 1, 1), padding=self.padding
        )
    
    def call(self, x):
        y = self.spatial_pooling(x)
        
        w = self.global_pooling(y)
        w = self.conv1(w)
        w = self.activation(w)
        w = self.conv2(w)
        w = tf.keras.activations.sigmoid(w)
        
        y = y * w
        y = self.temporal_pooling(y)
        
        return y
