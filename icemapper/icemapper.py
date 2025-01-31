import tensorflow as tf
import layers


def Conv3DResUNetEncoder(
    input_shape, n_steps=3, start_n_filters=32, dropout=0, mcdropout=False,
    pooling=tf.keras.layers.MaxPooling3D, name="Conv3DResUNetEncoder", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape, name="inputs")

    outputs = []
    x = inputs
    n_filters = start_n_filters
    for _ in range(n_steps):
        x = layers.ResidualWithProjection(
            layers.Conv3DBatchNormAct_x2(
                n_filters, spatial_dropout=dropout, mcdropout=mcdropout
            ),
            n_filters
        )(x)
        _, timesteps, height, width, n_features = x.shape
        output = pooling(pool_size=(timesteps, 1, 1))(x)
        output = tf.keras.layers.Reshape((height, width, n_features))(output)
        outputs.append(output)
        x = pooling(
            pool_size=(2 if timesteps >= 2 else 1, 2, 2),
            padding="same"
        )(x)
        n_filters *= 2

    x = layers.ResidualWithProjection(
        layers.Conv3DBatchNormAct_x2(
            n_filters, spatial_dropout=dropout, mcdropout=mcdropout
        ),
        n_filters
    )(x)
    timesteps = x.shape[1]
    _, timesteps, height, width, n_features = x.shape
    output = pooling(pool_size=(timesteps, 1, 1))(x)
    output = tf.keras.layers.Reshape((height, width, n_features))(output)
    outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def ResUNetDecoder(
    input_shape, n_outputs, n_steps=3, last_activation="softmax", dropout=0,
    mcdropout=False, name="ResUNetDecoder", **kwargs
):
    encoded_height, encoded_width, encoded_depth = input_shape
    inputs = []

    input1 = tf.keras.layers.Input(input_shape)
    inputs.append(input1)

    x = input1
    n_filters = encoded_depth // 2
    input_height, input_width = encoded_height * 2, encoded_width * 2
    for _ in range(n_steps):
        upsampling = layers.UpConv(n_filters)(x)
        input_i = tf.keras.layers.Input((input_height, input_width, n_filters))
        inputs.append(input_i)
        concat = tf.keras.layers.Concatenate()([upsampling, input_i])
        x = layers.ResidualWithProjection(
            layers.ConvBatchNormAct_x2(
                n_filters, spatial_dropout=dropout, mcdropout=mcdropout
            ),
            n_filters
        )(concat)
        n_filters //= 2
        input_height *= 2
        input_width *= 2

    outputs = tf.keras.layers.Dense(n_outputs, activation=last_activation)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model


def ICEmapper(
    input_shape, n_outputs, n_steps=3, start_n_filters=32, last_activation="softmax",
    pooling=tf.keras.layers.MaxPooling3D, dropout=0, mcdropout=False, name="ICEmapper", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape, name="inputs")
    encoded = Conv3DResUNetEncoder(
        input_shape,
        n_steps=n_steps,
        start_n_filters=start_n_filters,
        pooling=pooling,
        dropout=dropout,
        mcdropout=mcdropout,
    )(inputs)
    decoded = ResUNetDecoder(
        encoded[-1].shape[1:],
        n_outputs,
        n_steps=n_steps,
        last_activation=last_activation,
        dropout=dropout,
        mcdropout=mcdropout,
    )(encoded[::-1])

    model = tf.keras.models.Model(inputs=inputs, outputs=decoded, name=name, **kwargs)
    return model


def ICEmapper_v2(
    input_shape, n_outputs, name="ICEmapper_v2", **kwargs
):
    return ICEmapper(input_shape, n_outputs, pooling=layers.TemporalWeightedPooling, name=name, **kwargs)


# def ICEmapperMini(
#     input_shape, n_outputs, last_activation="softmax", pooling=tf.keras.layers.MaxPooling3D,
#     dropout=0, mcdropout=False, name="ICEmapperMini", **kwargs
# ):
#     return ICEmapper(
#         input_shape,
#         n_outputs,
#         n_steps=3,
#         start_n_filters=16,
#         pooling=pooling,
#         last_activation=last_activation,
#         dropout=dropout,
#         mcdropout=mcdropout,
#         name=name,
#         **kwargs
#     )


def ULSTMMini(
    input_shape, n_outputs, n_steps=2, last_activation="softmax", start_n_filters=32,
    dropout=0, mcdropout=False, pooling=tf.keras.layers.MaxPooling3D, name="ULSTM",
    **kwargs
):
    """
    This is a deprecated model that we tested in our preliminary experiments reported in IGARSS24
    (https://doi.org/10.1109/IGARSS53475.2024.10640676).
    We keep this implementation here for those who is interested in that methodology comparison.
    Yet, it is not integrated with the new model we have now (ICEmapper), and using it with the
    same CLI will require some basic coding.
    We do not proceed with this model as it does not provide convincing performance gains, while
    being more computationally demanding. 
    """
    inputs = tf.keras.layers.Input(input_shape, name="features")

    encoded = []
    x = inputs
    n_filters = start_n_filters
    for step in range(n_steps):
        if step > 0:
            res = x
            res = tf.keras.layers.Dense(n_filters, use_bias=False)(res)
        x = tf.keras.layers.Conv3D(n_filters, (1, 3, 3), padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x, output1, _, output2, _ = tf.keras.layers.Bidirectional(
            tf.keras.layers.ConvLSTM2D(n_filters // 2, (3, 3), padding="same", return_sequences=True, return_state=True),
            merge_mode="concat",
        )(x)
        output = tf.keras.layers.Concatenate()([output1, output2])
        encoded.append(output)

        if step > 0:
            x = x + res
        x = pooling(pool_size=(1, 2, 2), padding="same")(x)
        if dropout:
            x = tf.keras.layers.SpatialDropout3D(dropout)(x, training=mcdropout)
        n_filters *= 2

    res = x
    res = tf.keras.layers.Dense(n_filters, use_bias=False)(res)
    res = tf.reduce_mean(res, axis=1)
    x = tf.keras.layers.Conv3D(n_filters, (1, 3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.ConvLSTM2D(n_filters // 2, (3, 3), padding="same", return_sequences=False, return_state=False),
        merge_mode="concat",
    )(x)
    output = x + res
    encoded.append(output)

    decoder = ResUNetDecoder(
        input_shape=encoded[-1].shape[1:],
        n_outputs=n_outputs,
        n_steps=n_steps,
        last_activation=last_activation,
        dropout=dropout,
        mcdropout=mcdropout,
        name="ULSTMMiniDecoder",
    )
    outputs = decoder(encoded[::-1])

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model
