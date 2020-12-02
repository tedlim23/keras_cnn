from tensorflow import keras as K
IMAGE_SIZE = (224,224)

def get_mbv2(num_classes):
    base_model = K.applications.MobileNetV2(
        input_shape=IMAGE_SIZE+(3,),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling='avg',
    )

    x = base_model.output
    # let's add a fully-connected layer
    # x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = K.layers.Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = K.models.Model(inputs=base_model.input, outputs=predictions)

    return model

def get_resnet50(num_classes):
    base_model = K.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling='avg',
    )
    x = base_model.output
    # let's add a fully-connected layer
    # x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = K.layers.Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = K.models.Model(inputs=base_model.input, outputs=predictions)

    return model

def fine_tune(model, layer_num):
    for layer in model.layers[:layer_num]:
        layer.trainable = False
    for layer in model.layers[layer_num:]:
        layer.trainable = True
    return model

def compile_model(model, opt, loss):
    opt_options = {
        "adam" : K.optimizers.Adam(lr=0.0001),
        "sgd": K.optimizers.SGD(lr=0.005, momentum=0.9)
    }
    loss_options = {
        'binary': K.losses.BinaryCrossentropy(),
        'categorical': K.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0
        )
    }
    model.compile(
        optimizer= opt_options[opt],
        loss= loss_options[loss],
        metrics=['accuracy', K.metrics.Recall(class_id = 1)]
    )
    return model