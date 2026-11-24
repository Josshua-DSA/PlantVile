"""Models: a from-scratch CNN and transfer-learning backbones, plus compile()."""
import keras
import numpy as np
from keras import layers, regularizers


def _l2(cfg_model):
    weight = cfg_model["l2_weight_decay"]
    return regularizers.l2(weight) if weight > 0 else None


def build_simple_cnn(cfg_model, num_classes, input_shape):
    """Conv-BN-ReLU-MaxPool stack, doubling filters each block. Pure course material."""
    cfg = cfg_model
    reg = _l2(cfg)
    inputs = keras.Input(shape=input_shape)
    x = inputs
    filters = cfg["filters_base"]
    for block in range(cfg["conv_blocks"]):
        x = layers.Conv2D(filters, cfg["kernel_size"], padding="same",
                          kernel_regularizer=reg, name="conv%d" % (block + 1))(x)
        if cfg["batch_norm"]:
            x = layers.BatchNormalization(name="bn%d" % (block + 1))(x)
        x = layers.Activation("relu", name="relu%d" % (block + 1))(x)
        x = layers.MaxPooling2D(cfg["pool_size"], name="pool%d" % (block + 1))(x)
        filters *= 2

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if cfg["dense_units"] > 0:
        x = layers.Dense(cfg["dense_units"], kernel_regularizer=reg, name="dense")(x)
        if cfg["batch_norm"]:
            x = layers.BatchNormalization(name="bn_head")(x)
        x = layers.Activation("relu", name="relu_head")(x)
    if cfg["dropout"] > 0:
        x = layers.Dropout(cfg["dropout"], name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32",
                           name="predictions")(x)
    return keras.Model(inputs, outputs, name="simple_cnn")


def _backbone(arch, input_shape, weights):
    from keras import applications

    factory = {
        "vgg16": applications.VGG16,
        "resnet50": applications.ResNet50,
        "mobilenetv2": applications.MobileNetV2,
        "efficientnetb0": applications.EfficientNetB0,
    }[arch]
    return factory(include_top=False, weights=weights, input_shape=input_shape)


def build_transfer_model(cfg_model, num_classes, input_shape):
    """Frozen (or partially unfrozen) ImageNet backbone + classifier head."""
    cfg = cfg_model
    weights = "imagenet" if cfg["weights"] == "imagenet" else None
    base = _backbone(cfg["arch"], input_shape, weights)
    base.trainable = not cfg["freeze_base"]

    if cfg["freeze_base"] and cfg["fine_tune_at"] > 0:
        base.trainable = True
        cut = cfg["fine_tune_at"]
        for layer in base.layers[:cut]:
            layer.trainable = False
        # BatchNorm in a partially unfrozen backbone destabilises training
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    reg = _l2(cfg)
    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if cfg["dense_units"] > 0:
        x = layers.Dense(cfg["dense_units"], kernel_regularizer=reg, name="dense")(x)
        if cfg["batch_norm"]:
            x = layers.BatchNormalization(name="bn_head")(x)
        x = layers.Activation("relu", name="relu_head")(x)
    if cfg["dropout"] > 0:
        x = layers.Dropout(cfg["dropout"], name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32",
                           name="predictions")(x)
    return keras.Model(inputs, outputs, name=cfg["arch"])


def build_model(cfg, num_classes):
    """Dispatch on model.arch and print a one-line summary."""
    data, cfg_model = cfg["data"], cfg["model"]
    input_shape = (data["img_height"], data["img_width"], data["channels"])

    if cfg_model["arch"] == "simple_cnn":
        model = build_simple_cnn(cfg_model, num_classes, input_shape)
    else:
        model = build_transfer_model(cfg_model, num_classes, input_shape)

    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    frozen = sum(int(np.prod(w.shape)) for w in model.non_trainable_weights)
    print("\nModel: %s  input=%s  kelas=%d" % (model.name, input_shape, num_classes))
    print("  parameter trainable : %s" % f"{trainable:,}")
    print("  parameter frozen    : %s" % f"{frozen:,}")
    return model


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Categorical focal loss: down-weights easy examples in imbalanced sets."""

    def loss_fn(y_true, y_pred):
        y_true = keras.ops.cast(y_true, "float32")
        y_pred = keras.ops.cast(y_pred, "float32")
        y_pred = keras.ops.clip(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        cross_entropy = -y_true * keras.ops.log(y_pred)
        weight = alpha * keras.ops.power(1.0 - y_pred, gamma)
        return keras.ops.sum(weight * cross_entropy, axis=-1)

    return loss_fn


def _build_optimizer(cfg_train):
    name = cfg_train["optimizer"]
    common = {
        "learning_rate": cfg_train["learning_rate"],
        "epsilon": cfg_train["epsilon"],
    }
    if name == "sgd":
        return keras.optimizers.SGD(
            momentum=cfg_train["momentum"], nesterov=True, **common)
    if name == "adamw":
        return keras.optimizers.AdamW(
            beta_1=cfg_train["beta_1"], beta_2=cfg_train["beta_2"], **common)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(momentum=cfg_train["momentum"], **common)
    if name == "nadam":
        return keras.optimizers.Nadam(
            beta_1=cfg_train["beta_1"], beta_2=cfg_train["beta_2"], **common)
    return keras.optimizers.Adam(
        beta_1=cfg_train["beta_1"], beta_2=cfg_train["beta_2"], **common)


def _build_loss(cfg_train):
    name = cfg_train["loss"]
    smoothing = cfg_train["label_smoothing"]
    if name == "focal":
        return focal_loss()
    if name == "sparse_categorical_crossentropy":
        return keras.losses.SparseCategoricalCrossentropy()
    return keras.losses.CategoricalCrossentropy(label_smoothing=smoothing)


def compile_model(model, cfg_train):
    """Compile with the configured optimizer / loss / metrics."""
    model.compile(
        optimizer=_build_optimizer(cfg_train),
        loss=_build_loss(cfg_train),
        metrics=["accuracy"],
    )
    print("  optimizer=%s lr=%s loss=%s"
          % (cfg_train["optimizer"], cfg_train["learning_rate"], cfg_train["loss"]))
    return model
