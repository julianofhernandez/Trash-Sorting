# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# THIS FILE HAS BEEN EDITED FROM THE ORIGINAL


import copy
import math

import numpy as np

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast


DEFAULT_BLOCKS_ARGS = {
    "efficientnetv2-b3": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "truncated_normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1.0 / 3.0,
        "mode": "fan_out",
        "distribution": "uniform",
    },
}

def round_filters(filters, width_coefficient, min_depth, depth_divisor):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_filters = max(
        minimum_depth,
        int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def MBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name=None,
):
    """MBConv block: Mobile Inverted Residual Bottleneck."""
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if name is None:
        name = backend.get_uid("block0")

    def apply(inputs):
        # Expansion phase
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                data_format="channels_last",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(
                axis=bn_axis,
                momentum=bn_momentum,
                name=name + "expand_bn",
            )(x)
            x = layers.Activation(activation, name=name + "expand_activation")(
                x
            )
        else:
            x = inputs

        # Depthwise conv
        x = layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            depthwise_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            data_format="channels_last",
            use_bias=False,
            name=name + "dwconv2",
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis, momentum=bn_momentum, name=name + "bn"
        )(x)
        x = layers.Activation(activation, name=name + "activation")(x)

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
            if bn_axis == 1:
                se_shape = (filters, 1, 1)
            else:
                se_shape = (1, 1, filters)
            se = layers.Reshape(se_shape, name=name + "se_reshape")(se)

            se = layers.Conv2D(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = layers.Conv2D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)

            x = layers.multiply([x, se], name=name + "se_excite")

            # Output phase
            x = layers.Conv2D(
                filters=output_filters,
                kernel_size=1,
                strides=1,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                padding="same",
                data_format="channels_last",
                use_bias=False,
                name=name + "project_conv",
            )(x)
            x = layers.BatchNormalization(
                axis=bn_axis, momentum=bn_momentum, name=name + "project_bn"
            )(x)

            if strides == 1 and input_filters == output_filters:
                if survival_probability:
                    x = layers.Dropout(
                        survival_probability,
                        noise_shape=(None, 1, 1, 1),
                        name=name + "drop",
                    )(x)
                x = layers.add([x, inputs], name=name + "add")
        return x

    return apply


def FusedMBConvBlock(
    input_filters: int,
    output_filters: int,
    expand_ratio=1,
    kernel_size=3,
    strides=1,
    se_ratio=0.0,
    bn_momentum=0.9,
    activation="swish",
    survival_probability: float = 0.8,
    name=None,
):
    """Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a
    conv2d."""
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    if name is None:
        name = backend.get_uid("block0")

    def apply(inputs):
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            x = layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                data_format="channels_last",
                padding="same",
                use_bias=False,
                name=name + "expand_conv",
            )(inputs)
            x = layers.BatchNormalization(
                axis=bn_axis, momentum=bn_momentum, name=name + "expand_bn"
            )(x)
            x = layers.Activation(
                activation=activation, name=name + "expand_activation"
            )(x)
        else:
            x = inputs

        # Squeeze and excite
        if 0 < se_ratio <= 1:
            filters_se = max(1, int(input_filters * se_ratio))
            se = layers.GlobalAveragePooling2D(name=name + "se_squeeze")(x)
            if bn_axis == 1:
                se_shape = (filters, 1, 1)
            else:
                se_shape = (1, 1, filters)

            se = layers.Reshape(se_shape, name=name + "se_reshape")(se)

            se = layers.Conv2D(
                filters_se,
                1,
                padding="same",
                activation=activation,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_reduce",
            )(se)
            se = layers.Conv2D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer=CONV_KERNEL_INITIALIZER,
                name=name + "se_expand",
            )(se)

            x = layers.multiply([x, se], name=name + "se_excite")

        # Output phase:
        x = layers.Conv2D(
            output_filters,
            kernel_size=1 if expand_ratio != 1 else kernel_size,
            strides=1 if expand_ratio != 1 else strides,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            padding="same",
            use_bias=False,
            name=name + "project_conv",
        )(x)
        x = layers.BatchNormalization(
            axis=bn_axis, momentum=bn_momentum, name=name + "project_bn"
        )(x)
        if expand_ratio == 1:
            x = layers.Activation(
                activation=activation, name=name + "project_activation"
            )(x)

        # Residual:
        if strides == 1 and input_filters == output_filters:
            if survival_probability:
                x = layers.Dropout(
                    survival_probability,
                    noise_shape=(None, 1, 1, 1),
                    name=name + "drop",
                )(x)
            x = layers.add([x, inputs], name=name + "add")
        return x

    return apply


def EfficientNetV2(
    n_classes,
    drop_connect_rate=0.2,
    depth_divisor=8,
    min_depth=8,
    bn_momentum=0.9,
    activation="swish",
    blocks_args="default",
    model_name="efficientnetv2-b3",
    include_preprocessing=True,

    mode='training',
    l2_regularization=0.0005,
    min_scale=None,
    max_scale=None,
    scales=None,
    aspect_ratios_global=None,
    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                             [1.0, 2.0, 0.5],
                             [1.0, 2.0, 0.5]],
    two_boxes_for_ar1=True,
    steps=[8, 16, 32, 64, 100, 300],
    offsets=None,
    clip_boxes=False,
    variances=[0.1, 0.1, 0.2, 0.2],
    coords='centroids',
    normalize_coords=True,
    confidence_thresh=0.01,
    iou_threshold=0.45,
    top_k=200,
    nms_max_output_size=400,
    return_predictor_sizes=False
):
    width_coefficient = 1.2
    depth_coefficient = 1.4
    default_size = 300

    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS_ARGS[model_name]

    input_shape = (default_size, default_size, 3)
    
    # SSD stuff

    img_height, img_width, img_channels = input_shape
    n_predictor_layers = 6 # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    img_input = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    x = img_input

    if include_preprocessing:
        # Apply original V1 preprocessing for Bx variants
        # if number of channels allows it
        num_channels = input_shape[bn_axis - 1]
        if model_name.split("-")[-1].startswith("b") and num_channels == 3:
            x = layers.Rescaling(scale=1.0 / 255)(x)
            x = layers.Normalization(
                mean=[0.485, 0.456, 0.406],
                variance=[0.229**2, 0.224**2, 0.225**2],
                axis=bn_axis,
            )(x)
        else:
            x = layers.Rescaling(scale=1.0 / 128.0, offset=-1)(x)

    # Build stem
    stem_filters = round_filters(
        filters=blocks_args[0]["input_filters"],
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor,
    )
    x = layers.Conv2D(
        filters=stem_filters,
        kernel_size=3,
        strides=2,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(x)
    stride_counter = 1  # used to tell when to take info from a layer for SSD
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=bn_momentum,
        name="stem_bn",
    )(x)
    x = layers.Activation(activation, name="stem_activation")(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum(args["num_repeat"] for args in blocks_args))

    prepred_layers = []

    for (i, args) in enumerate(blocks_args):
        assert args["num_repeat"] > 0

        # Update block input and output filters based on depth multiplier.
        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor,
        )

        # Determine which conv type to use:
        block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient
        )
        for j in range(repeats):
            # The first block needs to take care of stride and filter size
            # increase.
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            if args["strides"] == 2:
                if stride_counter in [3, 4]:
                    prepred_layers.append(x)
                #print(i, j, stride_counter)
                stride_counter += 1

            x = block(
                activation=activation,
                bn_momentum=bn_momentum,
                survival_probability=drop_connect_rate * b / blocks,
                name="block{}{}_".format(i + 1, chr(j + 97)),
                **args,
            )(x)
            b += 1

    # SSD Shortcut 10
    prepred_layers.append(x)

    x = layers.Conv2D(128, (1, 1), activation=activation, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, name='extra_conv_1_1')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='extra_conv_1_padding')(x)
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), activation=activation, padding='valid', kernel_initializer=CONV_KERNEL_INITIALIZER, name='extra_conv_1_2')(x)

    # SSD Shortcut 5
    prepred_layers.append(x)

    x = layers.Conv2D(128, (1, 1), activation=activation, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, name='extra_conv_2_1')(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), activation=activation, padding='valid', kernel_initializer=CONV_KERNEL_INITIALIZER, name='extra_conv_2_2')(x)

    # SSD Shortcut 3
    prepred_layers.append(x)
    
    x = layers.Conv2D(128, (1, 1), activation=activation, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, name='extra_conv_3_1')(x)
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), activation=activation, padding='valid', kernel_initializer=CONV_KERNEL_INITIALIZER, name='extra_conv_3_2')(x)

    # SSD Shortcut 1
    prepred_layers.append(x)

    prepred_layers[0] = L2Normalization(gamma_init=20, name='prepred_norm')(prepred_layers[0])

    ### Build the convolutional predictor layers on top of the base network
    conv_pred_class_layers = []
    for i in range(6):
        x = layers.Conv2D(n_boxes[i] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=f'conv_pred_class_{i}')(prepred_layers[i])
        conv_pred_class_layers.append(x)
    conv_pred_loc_layers = []
    for i in range(6):
        x = layers.Conv2D(n_boxes[i] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name=f'conv_pred_loc_{i}')(prepred_layers[i])
        conv_pred_loc_layers.append(x)

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)
    anchors = []
    for i in range(6):
        x = AnchorBoxes(img_height, img_width, this_scale=scales[i], next_scale=scales[i + 1], aspect_ratios=aspect_ratios[i],
                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[i], this_offsets=offsets[i], clip_boxes=clip_boxes,
                        variances=variances, coords=coords, normalize_coords=normalize_coords, name=f'mbox_priorbox_{i}')(conv_pred_loc_layers[i])
        anchors.append(x)

    ### Reshape
    conv_pred_class_layers_reshape = []
    for i in range(6):
        x = layers.Reshape((-1, n_classes), name=f'conv_pred_class_reshape_{i}')(conv_pred_class_layers[i])
        conv_pred_class_layers_reshape.append(x)
    conv_pred_loc_layers_reshape = []
    for i in range(6):
        x = layers.Reshape((-1, 4), name=f'conv_pred_loc_reshape_{i}')(conv_pred_loc_layers[i])
        conv_pred_loc_layers_reshape.append(x)
    anchors_reshape = []
    for i in range(6):
        x = layers.Reshape((-1, 8), name=f'mbox_priorbox_reshape_{i}')(anchors[i])
        anchors_reshape.append(x)

    ### Concatenate the predictions from the different layers
    mbox_conf = layers.Concatenate(axis=1, name='mbox_conf')(conv_pred_class_layers_reshape)
    mbox_loc = layers.Concatenate(axis=1, name='mbox_loc')(conv_pred_loc_layers_reshape)
    mbox_priorbox = layers.Concatenate(axis=1, name='mbox_priorbox')(anchors_reshape)

    mbox_conf_softmax = layers.Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    predictions = layers.Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=img_input, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=img_input, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=img_input, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([model.get_layer('conv_pred_class_0').output_shape[1:3],
                                    model.get_layer('conv_pred_class_1').output_shape[1:3],
                                    model.get_layer('conv_pred_class_2').output_shape[1:3],
                                    model.get_layer('conv_pred_class_3').output_shape[1:3],
                                    model.get_layer('conv_pred_class_4').output_shape[1:3],
                                    model.get_layer('conv_pred_class_5').output_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
