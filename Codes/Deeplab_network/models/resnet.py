import tensorflow as tf
from tensorflow.keras import layers, Model

class ConvBN(layers.Layer):
    """Conv2D -> (optional) Bias -> BatchNorm -> (optional) ReLU"""
    def __init__(self, filters, kernel_size, stride=1, dilation=1,
                 use_bias=False, activation=None, bn_trainable=True, name=None):
        super().__init__(name=name)
        self.conv = layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=stride,
            padding="same", dilation_rate=dilation, use_bias=use_bias,
            kernel_initializer="he_normal", name="conv"
        )
        self.bn = layers.BatchNormalization(
            axis=-1, momentum=0.997, epsilon=1e-5, center=True, scale=True,
            trainable=bn_trainable, name="bn"
        )
        self.act = layers.ReLU(name="relu") if activation == "relu" else None

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x


class Bottleneck(layers.Layer):
    """
    ResNet-v1 bottleneck block (1x1 -> 3x3 -> 1x1, with optional projection).
    - If half_size=True, first 1x1 uses stride=2 (downsampling).
    - For dilated variant, 3x3 uses dilation>1 and stride=1.
    """
    def __init__(self, out_channels, half_size=False, identity_connection=True,
                 dilation=1, bn_trainable=True, name=None):
        super().__init__(name=name)
        assert out_channels % 4 == 0, "Bottleneck number of output ERROR!"
        mid = out_channels // 4
        stride1 = 2 if half_size else 1

        # Main branch
        self.conv1 = ConvBN(mid, 1, stride=stride1, activation="relu",
                            bn_trainable=bn_trainable, name="conv1")
        self.conv2 = ConvBN(mid, 3, stride=1, dilation=dilation, activation="relu",
                            bn_trainable=bn_trainable, name="conv2")
        self.conv3 = ConvBN(out_channels, 1, stride=1, activation=None,
                            bn_trainable=bn_trainable, name="conv3")

        # Shortcut
        self.use_proj = (not identity_connection) or half_size
        self.proj = (ConvBN(out_channels, 1, stride=stride1, activation=None,
                            bn_trainable=bn_trainable, name="shortcut")
                     if self.use_proj else None)

        self.add = layers.Add(name="add")
        self.out_relu = layers.ReLU(name="relu")

    def call(self, x, training=False):
        shortcut = x if self.proj is None else self.proj(x, training=training)
        out = self.conv1(x, training=training)
        out = self.conv2(out, training=training)
        out = self.conv3(out, training=training)
        out = self.add([shortcut, out])
        out = self.out_relu(out)
        return out


class ASPP(layers.Layer):
    """Parallel 3x3 dilated convs to num_classes, then sum."""
    def __init__(self, num_classes, dilations, name="decoder/aspp"):
        super().__init__(name=name)
        self.branches = []
        for i, d in enumerate(dilations, start=1):
            self.branches.append(
                layers.Conv2D(
                    filters=num_classes, kernel_size=3, strides=1, padding="same",
                    dilation_rate=d, use_bias=True, kernel_initializer="he_normal",
                    name=f"conv{i}"
                )
            )
        self.sum = layers.Add(name="sum")

    def call(self, x):
        outs = [conv(x) for conv in self.branches]
        return self.sum(outs)


class ResNetSegmentation(Model):
    """
    TF2.15 Keras reimplementation of the provided TF1.7 architecture (DeepLab-style backbone).
    """
    def __init__(self, num_classes, encoder_name='res50',
                 freeze_encoder_bn=True, name="resnet_segmentation"):
        super().__init__(name=name)
        if encoder_name not in ['res101', 'res50']:
            raise ValueError("encoder_name must be 'res50' or 'res101'")
        self.encoder_name = encoder_name
        bn_trainable = not freeze_encoder_bn

        # Stem
        self.stem = ConvBN(64, 7, stride=2, activation="relu",
                           bn_trainable=bn_trainable, name="resnet/conv1")
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="resnet/pool1")

        # Block 1 (3 units, out=256)
        self.b1_u1 = Bottleneck(256, half_size=False, identity_connection=False,
                                dilation=1, bn_trainable=bn_trainable, name="resnet/block1/unit_1")
        self.b1_u2 = Bottleneck(256, dilation=1, bn_trainable=bn_trainable, name="resnet/block1/unit_2")
        self.b1_u3 = Bottleneck(256, dilation=1, bn_trainable=bn_trainable, name="resnet/block1/unit_3")

        # Block 2 (4 units, out=512; first downsamples)
        self.b2_u1 = Bottleneck(512, half_size=True, identity_connection=False,
                                dilation=1, bn_trainable=bn_trainable, name="resnet/block2/unit_1")
        self.b2_u2 = Bottleneck(512, dilation=1, bn_trainable=bn_trainable, name="resnet/block2/unit_2")
        self.b2_u3 = Bottleneck(512, dilation=1, bn_trainable=bn_trainable, name="resnet/block2/unit_3")
        self.b2_u4 = Bottleneck(512, dilation=1, bn_trainable=bn_trainable, name="resnet/block2/unit_4")

        # Block 3 (res50: 6 units | res101: 23 units), dilation=2
        num_block3 = 23 if encoder_name == 'res101' else 6
        self.block3_units = []
        self.block3_units.append(
            Bottleneck(1024, identity_connection=False, dilation=2,
                       bn_trainable=bn_trainable, name="resnet/block3/unit_1")
        )
        for i in range(2, num_block3 + 1):
            self.block3_units.append(
                Bottleneck(1024, dilation=2, bn_trainable=bn_trainable,
                           name=f"resnet/block3/unit_{i}")
            )

        # Block 4 (3 units, dilation=4)
        self.b4_u1 = Bottleneck(2048, identity_connection=False, dilation=4,
                                bn_trainable=bn_trainable, name="resnet/block4/unit_1")
        self.b4_u2 = Bottleneck(2048, dilation=4, bn_trainable=bn_trainable, name="resnet/block4/unit_2")
        self.b4_u3 = Bottleneck(2048, dilation=4, bn_trainable=bn_trainable, name="resnet/block4/unit_3")

        # Decoder: ASPP
        self.aspp = ASPP(num_classes=num_classes, dilations=[6, 12, 18, 24], name="decoder/aspp")

    def encode(self, x, training=False):
        x = self.stem(x, training=training)
        x = self.pool1(x)
        x = self.b1_u1(x, training=training); x = self.b1_u2(x, training=training); x = self.b1_u3(x, training=training)
        x = self.b2_u1(x, training=training); x = self.b2_u2(x, training=training); x = self.b2_u3(x, training=training); x = self.b2_u4(x, training=training)
        for u in self.block3_units: x = u(x, training=training)
        x = self.b4_u1(x, training=training); x = self.b4_u2(x, training=training); x = self.b4_u3(x, training=training)
        return x

    def decode(self, x): return self.aspp(x)

    def call(self, inputs, training=False):
        return self.decode(self.encode(inputs, training=training))

