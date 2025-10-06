import tensorflow as tf
from tensorflow.keras import layers, Model

class Conv2D(layers.Layer):
    def __init__(self, filters, kernel_size, stride=1, dilation=1, use_bias=False, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kh = self.kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.dilation = dilation
        self.use_bias = use_bias

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        self.weights_var = self.add_weight(
            name="weights", shape=(self.kh, self.kw, in_ch, self.filters),
            initializer="he_normal", trainable=True
        )
        if self.use_bias:
            self.biases_var = self.add_weight(
                name="biases", shape=(self.filters,), initializer="zeros", trainable=True
            )

    def call(self, x):
        y = tf.nn.conv2d(
            x, self.weights_var,
            strides=[1, self.stride, self.stride, 1],
            padding="SAME",
            dilations=[1, self.dilation, self.dilation, 1],
        )
        if self.use_bias:
            y = tf.nn.bias_add(y, self.biases_var)
        return y
    
def BN(name, trainable=False):
    bn = tf.keras.layers.BatchNormalization(
        name=name, axis=-1, momentum=0.999, epsilon=1e-3,
        center=True, scale=True, fused=False, trainable=trainable
    )
    return bn

class ConvBN(layers.Layer):
    def __init__(self, filters, kernel, stride=1, dilation=1,
                 use_bias=False, activation=None, conv_name=None, bn_name=None, bn_trainable=False):
        super().__init__(name=conv_name)
        self.conv = Conv2D(filters, kernel, stride=stride, dilation=dilation,
                           use_bias=use_bias, name=conv_name)
        self.bn = BN(bn_name, trainable=bn_trainable)
        self.act = layers.ReLU(name=f"{bn_name}_relu") if activation == "relu" else None

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x) if self.act is not None else x


class Bottleneck(layers.Layer):
    """
    ResNet-v1 bottleneck block.
    Names mirror TF1-style scopes for easy mapping, while remaining TF2-native:
      - Convs: res{tag}_branch2a, res{tag}_branch2b, res{tag}_branch2c, (optional) res{tag}_branch1
      - BNs:   bn{tag}_branch2a, bn{tag}_branch2b, bn{tag}_branch2c, (optional) bn{tag}_branch1
    """
    def __init__(self, out_channels, tag, half_size=False, identity_connection=True,
                 dilation=1, bn_trainable=True):
        super().__init__(name=f"res{tag}")
        assert out_channels % 4 == 0, "Bottleneck number of output ERROR!"
        mid = out_channels // 4
        stride1 = 2 if half_size else 1

        # branch2
        self.b2a = ConvBN(
            mid, 1, stride=stride1, activation="relu",
            conv_name=f"res{tag}_branch2a", bn_name=f"bn{tag}_branch2a",
            bn_trainable=bn_trainable
        )
        self.b2b = ConvBN(
            mid, 3, stride=1, dilation=dilation, activation="relu",
            conv_name=f"res{tag}_branch2b", bn_name=f"bn{tag}_branch2b",
            bn_trainable=bn_trainable
        )
        self.b2c = ConvBN(
            out_channels, 1, stride=1, activation=None,
            conv_name=f"res{tag}_branch2c", bn_name=f"bn{tag}_branch2c",
            bn_trainable=bn_trainable
        )

        # branch1 (projection) if needed
        self.use_proj = (not identity_connection) or half_size
        if self.use_proj:
            self.proj = ConvBN(
                out_channels, 1, stride=stride1, activation=None,
                conv_name=f"res{tag}_branch1", bn_name=f"bn{tag}_branch1",
                bn_trainable=bn_trainable
            )
        else:
            self.proj = None

        self.add = layers.Add(name=f"res{tag}")
        self.relu = layers.ReLU(name=f"res{tag}_relu")

    def call(self, x, training=False):
        shortcut = x if self.proj is None else self.proj(x, training=training)
        out = self.b2a(x, training=training)
        out = self.b2b(out, training=training)
        out = self.b2c(out, training=training)
        out = self.add([shortcut, out])
        out = self.relu(out)
        return out


class ASPPAdd(layers.Layer):
    def __init__(self, num_classes, dilations=[6, 12, 18, 24]):
        super().__init__(name="aspp")  # wrapper has a different name
        self.branches = [
            Conv2D(num_classes, 3, stride=1, dilation=d, use_bias=True, name=f"fc1_voc12_c{i}")
            for i, d in enumerate(dilations)
        ]
        self.sum = layers.Add(name="fc1_voc12")  # TF1 add node name

    def call(self, x):
        return self.sum([b(x) for b in self.branches])

# ----------------- DeepLabV2 Model -----------------

class DeepLabV2(Model):
    """
    DeepLab v2 (ResNet-v1-101 style) in TF2 with clean, TF2-native naming.
    """
    def __init__(self, num_classes, freeze_bn=True, name="deeplab_v2"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.bn_trainable = not freeze_bn

        self.conv1 = Conv2D(64, 7, stride=2, use_bias=False, name="conv1")
        self.bn_conv1 = BN("bn_conv1", trainable=self.bn_trainable)
        self.relu_conv1 = layers.ReLU(name="bn_conv1_relu")
        self.pool1 = layers.MaxPool2D(pool_size=3, strides=2, padding="same", name="pool1")

        # Block 2 (256)
        self.res2a = Bottleneck(256, "2a", half_size=False, identity_connection=False,
                                  dilation=1, bn_trainable=self.bn_trainable)
        self.res2b = Bottleneck(256, "2b", dilation=1, bn_trainable=self.bn_trainable)
        self.res2c = Bottleneck(256, "2c", dilation=1, bn_trainable=self.bn_trainable)

        # Block 3 (512), res3a downsamples
        self.res3a  = Bottleneck(512, "3a", half_size=True, identity_connection=False,
                                   dilation=1, bn_trainable=self.bn_trainable)
        self.res3b1 = Bottleneck(512, "3b1", dilation=1, bn_trainable=self.bn_trainable)
        self.res3b2 = Bottleneck(512, "3b2", dilation=1, bn_trainable=self.bn_trainable)
        self.res3b3 = Bottleneck(512, "3b3", dilation=1, bn_trainable=self.bn_trainable)

        # Block 4 (1024), dilation=2
        self.res4a = Bottleneck(1024, "4a", identity_connection=False, dilation=2,
                                  bn_trainable=self.bn_trainable)
        # Properly register res4b1..res4b22
        self.res4b = []
        for i in range(1, 23):  # 1..22
            block = Bottleneck(1024, f"4b{i}", dilation=2, bn_trainable=self.bn_trainable)
            self.res4b.append(block)
            setattr(self, f"res4b{i}", block)

        # Block 5 (2048), dilation=4
        self.res5a = Bottleneck(2048, "5a", identity_connection=False, dilation=4,
                                  bn_trainable=self.bn_trainable)
        self.res5b = Bottleneck(2048, "5b", dilation=4, bn_trainable=self.bn_trainable)
        self.res5c = Bottleneck(2048, "5c", dilation=4, bn_trainable=self.bn_trainable)

        # ASPP (sum) with exact TF1-style outer scope/branch names
        self.aspp = ASPPAdd(num_classes=num_classes, dilations=[6, 12, 18, 24])

    # Encoder (as TF1)
    def encode(self, x, training=False):
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = self.relu_conv1(x)
        x = self.pool1(x)

        x = self.res2a(x, training=training)
        x = self.res2b(x, training=training)
        x = self.res2c(x, training=training)

        x = self.res3a(x, training=training)
        x = self.res3b1(x, training=training)
        x = self.res3b2(x, training=training)
        x = self.res3b3(x, training=training)

        x = self.res4a(x, training=training)
        for block in self.res4b:
            x = block(x, training=training)

        x = self.res5a(x, training=training)
        x = self.res5b(x, training=training)
        x = self.res5c(x, training=training)
        return x

    # Decoder
    def decode(self, x):
        return self.aspp(x)

    def call(self, inputs, training=False):
        enc = self.encode(inputs, training=training)
        logits = self.decode(enc)
        return logits