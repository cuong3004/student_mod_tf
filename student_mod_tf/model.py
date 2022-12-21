

# import tensorflow as tf
# from tensorflow import keras
# from keras_cv_attention_models.attention_layers import (
#     batchnorm_with_activation,
#     conv2d_no_bias,
#     ChannelAffine,
#     depthwise_conv2d_no_bias,
#     drop_block,
#     drop_connect_rates_split,
#     group_norm,
#     layer_norm,
#     make_divisible,
#     mlp_block,
#     multi_head_self_attention,
#     output_block,
#     add_pre_post_process,
# )
# from keras_cv_attention_models.download_and_load import reload_model_weights
# import tensorflow_models as tfm

# # tfm.vision.layers import SqueezeExcitation

# LAYER_NORM_EPSILON = 1e-5
# PRETRAINED_DICT = {
#     "mobilevit_xxs": {"imagenet": "f9d1d4f7329726b4bb9069cd631a152e"},
#     "mobilevit_xs": {"imagenet": "dcd8591668946aa4ddb0159dfe65cc2b"},
#     "mobilevit_s": {"imagenet": "55f1051035ecd14e90ae3df80fd0c7f3"},
#     "mobilevit_v2_050": {"imagenet": {256: "a842a40c0f49dc2bbe935493caed061b"}},
#     "mobilevit_v2_075": {"imagenet": {256: "8588b3d6bf4aa750766ddc6d01824c67"}},
#     "mobilevit_v2_100": {"imagenet": {256: "55d499bbc29f0f6379a4cc6f610e10e8"}},
#     "mobilevit_v2_125": {"imagenet": {256: "b8af7b7668774796530f19dd5b6080fb"}},
#     "mobilevit_v2_150": {
#         "imagenet": {256: "065e7a07f7e2d0d74a33913195df9044"},
#         "imagenet22k": {256: "cf3c4ec278154ece62e8967faa5c0391", 384: "cdcfaebb573f8cd1f41044ac0e958204"},
#     },
#     "mobilevit_v2_175": {
#         "imagenet": {256: "627719428c6cb35f071a7ea69a6961c4"},
#         "imagenet22k": {256: "b849708a6b2c1f115b8b8c366e1d1a19", 384: "d3feef5108b6195d1c5525fb185bf720"},
#     },
#     "mobilevit_v2_200": {
#         "imagenet": {256: "1fe59d8bb2662761084d1c04259a778d"},
#         "imagenet22k": {256: "931f0be1761bcf8443359ec1661bb6a7", 384: "1dc6cdafb187611e5a4819272d64fba7"},
#     },
# }


# def bottle_in_linear_out_block(inputs, out_channel, strides=1, expand_ratio=4, use_shortcut=False, drop_rate=0, activation="swish", name=""):
#     hidden_dim = int(inputs.shape[-1] * expand_ratio)
#     deep = conv2d_no_bias(inputs, hidden_dim, kernel_size=1, strides=1, name=name + "deep_1_")
#     deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_1_")
#     deep = depthwise_conv2d_no_bias(deep, kernel_size=3, strides=strides, padding="SAME", name=name + "deep_2_")
#     deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_2_")
#     deep = conv2d_no_bias(deep, out_channel, kernel_size=1, strides=1, name=name + "deep_3_")
#     deep = batchnorm_with_activation(deep, activation=None, name=name + "deep_3_")
#     deep = drop_block(deep, drop_rate=drop_rate, name=name + "deep_")

#     out = keras.layers.Add()([inputs, deep]) if use_shortcut else deep
#     return out


# def linear_self_attention(inputs, qkv_bias=False, out_bias=False, attn_axis=2, attn_dropout=0, name=None):
#     input_channel = inputs.shape[-1]
#     qkv = conv2d_no_bias(inputs, 1 + input_channel * 2, kernel_size=1, use_bias=qkv_bias, name=name and name + "qkv_")
#     query, key, value = tf.split(qkv, [1, input_channel, input_channel], axis=-1)
#     context_score = keras.layers.Softmax(axis=attn_axis, name=name and name + "attention_scores")(query)  # on patch_hh * patch_ww dimension
#     context_score = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(context_score) if attn_dropout > 0 else context_score
#     # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {context_score.shape = }")

#     context_vector = keras.layers.Multiply()([key, context_score])  # [batch, height, width, input_channel]
#     context_vector = tf.reduce_sum(context_vector, keepdims=True, axis=attn_axis)  # on patch_hh * patch_ww dimension

#     out = tf.nn.relu(value) * context_vector
#     out = conv2d_no_bias(out, input_channel, kernel_size=1, use_bias=out_bias, name=name and name + "output")
#     return out


# def mhsa_mlp_block(
#     inputs,
#     out_channel,
#     num_heads=4,
#     qkv_bias=True,
#     mlp_ratio=4,
#     num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
#     use_linear_attention=False,  # False for V1, True for V2
#     use_conv_mlp=False,  # False for V1, True for V2
#     mlp_drop_rate=0,
#     attn_drop_rate=0,
#     drop_rate=0,
#     layer_scale=-1,
#     activation="gelu",
#     name=None,
# ):
#     attn = group_norm(inputs, groups=num_norm_groups, name=name + "attn_") if num_norm_groups > 0 else layer_norm(inputs, name=name + "attn_")
#     if use_linear_attention:  # V2
#         if num_norm_groups > 0:
#             attn = keras.layers.Reshape(attn.shape[1:])(attn)  # Or will throw error when converting tflite, if GroupNorm is followed by Conv2D
#         attn = linear_self_attention(attn, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
#     else:  # V1
#         attn = multi_head_self_attention(attn, num_heads, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
#     attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name and name + "1_gamma")(attn) if layer_scale >= 0 else attn
#     attn = drop_block(attn, drop_rate=drop_rate, name=name and name + "attn_")
#     attn_out = keras.layers.Add(name=name and name + "attn_out")([inputs, attn])

#     mlp = group_norm(attn_out, groups=num_norm_groups, name=name + "mlp_") if num_norm_groups > 0 else layer_norm(attn_out, name=name + "mlp_")
#     if use_conv_mlp and num_norm_groups > 0:  # V2
#         mlp = keras.layers.Reshape(mlp.shape[1:])(mlp)  # Or will throw error when converting tflite, if GroupNorm is followed by Conv2D
#     mlp = mlp_block(mlp, int(out_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=use_conv_mlp, activation=activation, name=name and name + "mlp_")
#     mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name and name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
#     mlp = drop_block(mlp, drop_rate=drop_rate, name=name and name + "mlp_")
#     return keras.layers.Add(name=name and name + "out")([attn_out, mlp])


# def transformer_pre_process(inputs, out_channel, patch_size=2, resize_first=False, use_depthwise=False, patches_to_batch=True, activation="swish", name=""):
#     nn = inputs

#     if resize_first:  # V2
#         patch_hh, patch_ww = int(tf.math.ceil(nn.shape[1] / patch_size)), int(tf.math.ceil(nn.shape[2] / patch_size))
#         # print(f"transformer_pre_process before resize: {nn.shape = }")
#         if patch_hh * patch_size != nn.shape[1] or patch_ww * patch_size != nn.shape[2]:
#             nn = tf.image.resize(nn, [patch_hh * patch_size, patch_ww * patch_size], method="bilinear")

#     if use_depthwise:  # V2
#         nn = depthwise_conv2d_no_bias(nn, kernel_size=3, strides=1, padding="SAME", name=name + "pre_1_")
#     else:  # V1
#         nn = conv2d_no_bias(nn, nn.shape[-1], kernel_size=3, strides=1, padding="SAME", name=name + "pre_1_")
#     nn = batchnorm_with_activation(nn, activation=activation, name=name + "pre_1_")
#     nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "pre_2_")

#     if not resize_first:  # V1
#         patch_hh, patch_ww = int(tf.math.ceil(nn.shape[1] / patch_size)), int(tf.math.ceil(nn.shape[2] / patch_size))
#         # print(f"transformer_pre_process before resize: {nn.shape = }")
#         if patch_hh * patch_size != nn.shape[1] or patch_ww * patch_size != nn.shape[2]:
#             nn = tf.image.resize(nn, [patch_hh * patch_size, patch_ww * patch_size], method="bilinear")

#     # Extract patchs, limit transpose permute length <= 4
#     # [batch, height, width, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch * 4, height // 2, width // 2, channel]
#     # print(f"transformer_pre_process after resize: {nn.shape = }")
#     nn = tf.reshape(nn, [-1, patch_ww, patch_size, out_channel])  # [batch * patch_hh * h_patch_size, patch_ww, w_patch_size, channel]
#     nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_hh * h_patch_size, w_patch_size, patch_ww, channel]
#     nn = tf.reshape(nn, [-1, patch_hh, patch_size * patch_size, patch_ww * out_channel])  # [batch, patch_hh, h_patch_size * w_patch_size, patch_ww * channel]
#     nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch, h_patch_size * w_patch_size, patch_hh, patch_ww * channel]
#     extract_shape = [-1, patch_hh, patch_ww, out_channel] if patches_to_batch else [-1, patch_size * patch_size, patch_hh * patch_ww, out_channel]
#     nn = tf.reshape(nn, extract_shape)

#     return nn


# def transformer_post_process(inputs, pre_attn, out_channel, patch_size=2, patch_height=-1, activation="swish", name=""):
#     if patch_height == -1:  # V1, [batch * 4, height // 2, width // 2, channel]
#         patch_hh, patch_ww, channel = inputs.shape[1], inputs.shape[2], inputs.shape[-1]
#     else:  # V2, [batch, 4, height // 2 * width // 2, channel]
#         patch_hh, patch_ww, channel = patch_height, inputs.shape[2] // patch_height, inputs.shape[-1]
#     # print(f"{patch_hh = }, {patch_ww = }, {channel = }, {inputs.shape = }")

#     # [batch * 4, height // 2, width // 2, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch, height, width, channel]
#     nn = tf.reshape(inputs, [-1, patch_size * patch_size, patch_hh, patch_ww * channel])  # [batch, h_patch_size * w_patch_size, patch_hh, patch_ww * channel]
#     nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch, patch_hh, h_patch_size * w_patch_size, patch_ww * channel]
#     nn = tf.reshape(nn, [-1, patch_size, patch_ww, channel])  # [batch * patch_hh * h_patch_size, w_patch_size, patch_ww, channel]
#     nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_hh * h_patch_size, patch_ww, w_patch_size, channel]
#     nn = tf.reshape(nn, [-1, patch_hh * patch_size, patch_ww * patch_size, channel])
#     # print(f"transformer_post_process before resize: {nn.shape = }")
#     if pre_attn is not None and (nn.shape[1] != pre_attn.shape[1] or nn.shape[2] != pre_attn.shape[2]):
#         nn = tf.image.resize(nn, [pre_attn.shape[1], pre_attn.shape[2]], method="bilinear")
#     # print(f"transformer_post_process after resize: {nn.shape = }")

#     nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "post_1_")
#     nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_1_")
#     if pre_attn is not None:  # V1
#         nn = tf.concat([pre_attn, nn], axis=-1)
#         nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=1, padding="SAME", name=name + "post_2_")
#         nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_2_")
#     return keras.layers.Activation("linear", name=name + "output")(nn)  # Identity, Just need a name here

# def se_attention(inputs, name, reduction=4):
#     b, h, w, c = inputs.shape
#     sq = keras.layers.GlobalAveragePooling2D(name="SE_pool_"+name)(inputs)
#     sq = tf.reshape(sq, (-1,c))
#     fc = keras.layers.Dense(c // reduction, use_bias=False, name="SE_ds1_"+name)(sq)
#     fc = tf.nn.swish(fc)
#     fc = keras.layers.Dense(c, use_bias=False, name="SE_ds2_"+name)(fc)
#     fc = tf.reshape(fc, (-1,1,1,c))

#     # print( inputs.shape)
#     if b == None:
#         return inputs * tf.broadcast_to(fc, (-1,h,w,c))
#     else:
#         inputs * tf.broadcast_to(fc, (b,h,w,c))
#     # return inputs * tf.broadcast_to(fc, (-1,h,w,c))

# def MobileViT(
#     num_blocks=[1, 3, 3, 5, 4],
#     out_channels=[32, 64, 96, 128, 160],
#     attn_channels=[0, 0, 144, 192, 240],  # Can be a list matching out_channels, or a float number for expansion ratio of out_channels
#     block_types=["conv", "conv", "transform", "transform", "transform"],
#     strides=[1, 2, 2, 2, 2],
#     expand_ratio=4,
#     stem_width=16,
#     patch_size=2,
#     patches_to_batch=True,  # True for V1, False for V2
#     resize_first=False,  # False for V1, True for V2
#     use_depthwise=False,  # False for V1, True for V2
#     use_fusion=True,  # True for V1, False for V2
#     num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
#     use_linear_attention=False,  # False for V1, True for V2
#     use_conv_mlp=False,  # False for V1, True for V2
#     output_num_features=640,
#     layer_scale=-1,
#     input_shape=(256, 256, 3),
#     num_classes=1000,
#     activation="swish",
#     drop_connect_rate=0,
#     classifier_activation="softmax",
#     dropout=0,
#     pretrained=None,
#     model_name="mobilevit",
#     kwargs=None,
# ):
#     inputs = keras.layers.Input(input_shape)
#     nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=1, padding="same", name="stem_")
#     nn = batchnorm_with_activation(nn, activation=activation, name="stem_")

#     # Save line width
#     mhsa_mlp_block_common_kwargs = {
#         "num_heads": 4,
#         "qkv_bias": True,
#         "mlp_ratio": 2,
#         "num_norm_groups": num_norm_groups,
#         "use_linear_attention": use_linear_attention,
#         "use_conv_mlp": use_conv_mlp,
#         "activation": activation,
#     }

#     """ stages """
#     total_blocks = sum(num_blocks)
#     global_block_id = 0
#     post_activation = activation if use_fusion else None
#     for id, (num_block, out_channel, block_type, stride) in enumerate(zip(num_blocks, out_channels, block_types, strides)):
#         stack_name = "stack{}_".format(id + 1)
#         is_conv_block = True if block_type[0].lower() == "c" else False
#         attn_channel = attn_channels[id] if isinstance(attn_channels, (list, tuple)) else make_divisible(attn_channels * out_channel, divisor=8)
#         for block_id in range(num_block):
#             name = stack_name + "block{}_".format(block_id + 1)
#             stride = stride if block_id == 0 else 1
#             use_shortcut = False if stride != 1 or nn.shape[-1] != out_channel else True
#             block_drop_rate = drop_connect_rate * global_block_id / total_blocks
#             global_block_id += 1
#             if is_conv_block or block_id == 0:  # First transformer block is also a conv block .
#                 nn = bottle_in_linear_out_block(nn, out_channel, stride, expand_ratio, use_shortcut, block_drop_rate, activation=activation, name=name)
#             else:
#                 if block_id == 1:  # pre
#                     pre_attn = nn if use_fusion else None
#                     patch_height = -1 if patches_to_batch else int(tf.math.ceil(nn.shape[1] / patch_size))
#                     nn = transformer_pre_process(nn, attn_channel, patch_size, resize_first, use_depthwise, patches_to_batch, activation=activation, name=name)
#                 nn = mhsa_mlp_block(nn, attn_channel, layer_scale=layer_scale, **mhsa_mlp_block_common_kwargs, name=name)
#                 # nn = se_attention(nn, name)
#                 nn = tfm.vision.layers.SqueezeExcitation(attn_channel, attn_channel, 0.25)(nn)
#                 if block_id == num_block - 1:  # post
#                     nn = group_norm(nn, groups=num_norm_groups, name=name + "post_") if num_norm_groups > 0 else layer_norm(nn, name=name + "post_")
#                     nn = transformer_post_process(nn, pre_attn, out_channel, patch_size, patch_height, activation=post_activation, name=name)
#     nn = output_block(nn, output_num_features, activation, num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
#     model = keras.models.Model(inputs, nn, name=model_name)
#     add_pre_post_process(model, rescale_mode="raw01")
#     reload_model_weights(model, PRETRAINED_DICT, "mobilevit", pretrained)
#     return model


# def MobileViT_XXS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
#     num_blocks = [1, 3, 3, 5, 4]
#     out_channels = [16, 24, 48, 64, 80]
#     attn_channels = [0, 0, 64, 80, 96]
#     output_num_features = 320
#     expand_ratio = 2
#     return MobileViT(**locals(), model_name="mobilevit_xxs", **kwargs)


# def MobileViT_XS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
#     num_blocks = [1, 3, 3, 5, 4]
#     out_channels = [32, 48, 64, 80, 96]
#     attn_channels = 1.5
#     output_num_features = 384
#     return MobileViT(**locals(), model_name="mobilevit_xs", **kwargs)


# def MobileViT_S(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
#     num_blocks = [1, 3, 3, 5, 4]
#     out_channels = [32, 64, 96, 128, 160]
#     attn_channels = 1.5
#     return MobileViT(**locals(), model_name="mobilevit_s", **kwargs)


# # import tensorflow as tf
# # from tensorflow import keras
# # from keras_cv_attention_models.attention_layers import (
# #     batchnorm_with_activation,
# #     conv2d_no_bias,
# #     ChannelAffine,
# #     depthwise_conv2d_no_bias,
# #     drop_block,
# #     drop_connect_rates_split,
# #     group_norm,
# #     layer_norm,
# #     make_divisible,
# #     mlp_block,
# #     multi_head_self_attention,
# #     output_block,
# #     add_pre_post_process,
# # )
# # from keras_cv_attention_models.download_and_load import reload_model_weights

# # LAYER_NORM_EPSILON = 1e-5
# # PRETRAINED_DICT = {
# #     "mobilevit_xxs": {"imagenet": "f9d1d4f7329726b4bb9069cd631a152e"},
# #     "mobilevit_xs": {"imagenet": "dcd8591668946aa4ddb0159dfe65cc2b"},
# #     "mobilevit_s": {"imagenet": "55f1051035ecd14e90ae3df80fd0c7f3"},
# #     "mobilevit_v2_050": {"imagenet": {256: "a842a40c0f49dc2bbe935493caed061b"}},
# #     "mobilevit_v2_075": {"imagenet": {256: "8588b3d6bf4aa750766ddc6d01824c67"}},
# #     "mobilevit_v2_100": {"imagenet": {256: "55d499bbc29f0f6379a4cc6f610e10e8"}},
# #     "mobilevit_v2_125": {"imagenet": {256: "b8af7b7668774796530f19dd5b6080fb"}},
# #     "mobilevit_v2_150": {
# #         "imagenet": {256: "065e7a07f7e2d0d74a33913195df9044"},
# #         "imagenet22k": {256: "cf3c4ec278154ece62e8967faa5c0391", 384: "cdcfaebb573f8cd1f41044ac0e958204"},
# #     },
# #     "mobilevit_v2_175": {
# #         "imagenet": {256: "627719428c6cb35f071a7ea69a6961c4"},
# #         "imagenet22k": {256: "b849708a6b2c1f115b8b8c366e1d1a19", 384: "d3feef5108b6195d1c5525fb185bf720"},
# #     },
# #     "mobilevit_v2_200": {
# #         "imagenet": {256: "1fe59d8bb2662761084d1c04259a778d"},
# #         "imagenet22k": {256: "931f0be1761bcf8443359ec1661bb6a7", 384: "1dc6cdafb187611e5a4819272d64fba7"},
# #     },
# # }


# # def bottle_in_linear_out_block(inputs, out_channel, strides=1, expand_ratio=4, use_shortcut=False, drop_rate=0, activation="swish", name=""):
# #     hidden_dim = int(inputs.shape[-1] * expand_ratio)
# #     deep = conv2d_no_bias(inputs, hidden_dim, kernel_size=1, strides=1, name=name + "deep_1_")
# #     deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_1_")
# #     deep = depthwise_conv2d_no_bias(deep, kernel_size=3, strides=strides, padding="SAME", name=name + "deep_2_")
# #     deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_2_")
# #     deep = conv2d_no_bias(deep, out_channel, kernel_size=1, strides=1, name=name + "deep_3_")
# #     deep = batchnorm_with_activation(deep, activation=None, name=name + "deep_3_")
# #     deep = drop_block(deep, drop_rate=drop_rate, name=name + "deep_")

# #     out = keras.layers.Add()([inputs, deep]) if use_shortcut else deep
# #     return out


# # def linear_self_attention(inputs, qkv_bias=False, out_bias=False, attn_axis=2, attn_dropout=0, name=None):
# #     input_channel = inputs.shape[-1]
# #     qkv = conv2d_no_bias(inputs, 1 + input_channel * 2, kernel_size=1, use_bias=qkv_bias, name=name and name + "qkv_")
# #     query, key, value = tf.split(qkv, [1, input_channel, input_channel], axis=-1)
# #     context_score = keras.layers.Softmax(axis=attn_axis, name=name and name + "attention_scores")(query)  # on patch_hh * patch_ww dimension
# #     context_score = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(context_score) if attn_dropout > 0 else context_score
# #     # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {context_score.shape = }")

# #     context_vector = keras.layers.Multiply()([key, context_score])  # [batch, height, width, input_channel]
# #     context_vector = tf.reduce_sum(context_vector, keepdims=True, axis=attn_axis)  # on patch_hh * patch_ww dimension

# #     out = tf.nn.relu(value) * context_vector
# #     out = conv2d_no_bias(out, input_channel, kernel_size=1, use_bias=out_bias, name=name and name + "output")
# #     return out


# # def mhsa_mlp_block(
# #     inputs,
# #     out_channel,
# #     num_heads=4,
# #     qkv_bias=True,
# #     mlp_ratio=4,
# #     num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
# #     use_linear_attention=False,  # False for V1, True for V2
# #     use_conv_mlp=False,  # False for V1, True for V2
# #     mlp_drop_rate=0,
# #     attn_drop_rate=0,
# #     drop_rate=0,
# #     layer_scale=-1,
# #     activation="gelu",
# #     name=None,
# # ):
# #     attn = group_norm(inputs, groups=num_norm_groups, name=name + "attn_") if num_norm_groups > 0 else layer_norm(inputs, name=name + "attn_")
# #     if use_linear_attention:  # V2
# #         if num_norm_groups > 0:
# #             attn = keras.layers.Reshape(attn.shape[1:])(attn)  # Or will throw error when converting tflite, if GroupNorm is followed by Conv2D
# #         attn = linear_self_attention(attn, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
# #     else:  # V1
# #         attn = multi_head_self_attention(attn, num_heads, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
# #     attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name and name + "1_gamma")(attn) if layer_scale >= 0 else attn
# #     attn = drop_block(attn, drop_rate=drop_rate, name=name and name + "attn_")
# #     attn_out = keras.layers.Add(name=name and name + "attn_out")([inputs, attn])

# #     mlp = group_norm(attn_out, groups=num_norm_groups, name=name + "mlp_") if num_norm_groups > 0 else layer_norm(attn_out, name=name + "mlp_")
# #     if use_conv_mlp and num_norm_groups > 0:  # V2
# #         mlp = keras.layers.Reshape(mlp.shape[1:])(mlp)  # Or will throw error when converting tflite, if GroupNorm is followed by Conv2D
# #     mlp = mlp_block(mlp, int(out_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=use_conv_mlp, activation=activation, name=name and name + "mlp_")
# #     mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name and name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
# #     mlp = drop_block(mlp, drop_rate=drop_rate, name=name and name + "mlp_")
# #     return keras.layers.Add(name=name and name + "out")([attn_out, mlp])


# # def transformer_pre_process(inputs, out_channel, patch_size=2, resize_first=False, use_depthwise=False, patches_to_batch=True, activation="swish", name=""):
# #     nn = inputs

# #     if resize_first:  # V2
# #         patch_hh, patch_ww = int(tf.math.ceil(nn.shape[1] / patch_size)), int(tf.math.ceil(nn.shape[2] / patch_size))
# #         # print(f"transformer_pre_process before resize: {nn.shape = }")
# #         if patch_hh * patch_size != nn.shape[1] or patch_ww * patch_size != nn.shape[2]:
# #             nn = tf.image.resize(nn, [patch_hh * patch_size, patch_ww * patch_size], method="bilinear")

# #     if use_depthwise:  # V2
# #         nn = depthwise_conv2d_no_bias(nn, kernel_size=3, strides=1, padding="SAME", name=name + "pre_1_")
# #     else:  # V1
# #         nn = conv2d_no_bias(nn, nn.shape[-1], kernel_size=3, strides=1, padding="SAME", name=name + "pre_1_")
# #     nn = batchnorm_with_activation(nn, activation=activation, name=name + "pre_1_")
# #     nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "pre_2_")

# #     if not resize_first:  # V1
# #         patch_hh, patch_ww = int(tf.math.ceil(nn.shape[1] / patch_size)), int(tf.math.ceil(nn.shape[2] / patch_size))
# #         # print(f"transformer_pre_process before resize: {nn.shape = }")
# #         if patch_hh * patch_size != nn.shape[1] or patch_ww * patch_size != nn.shape[2]:
# #             nn = tf.image.resize(nn, [patch_hh * patch_size, patch_ww * patch_size], method="bilinear")

# #     # Extract patchs, limit transpose permute length <= 4
# #     # [batch, height, width, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch * 4, height // 2, width // 2, channel]
# #     # print(f"transformer_pre_process after resize: {nn.shape = }")
# #     nn = tf.reshape(nn, [-1, patch_ww, patch_size, out_channel])  # [batch * patch_hh * h_patch_size, patch_ww, w_patch_size, channel]
# #     nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_hh * h_patch_size, w_patch_size, patch_ww, channel]
# #     nn = tf.reshape(nn, [-1, patch_hh, patch_size * patch_size, patch_ww * out_channel])  # [batch, patch_hh, h_patch_size * w_patch_size, patch_ww * channel]
# #     nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch, h_patch_size * w_patch_size, patch_hh, patch_ww * channel]
# #     extract_shape = [-1, patch_hh, patch_ww, out_channel] if patches_to_batch else [-1, patch_size * patch_size, patch_hh * patch_ww, out_channel]
# #     nn = tf.reshape(nn, extract_shape)

# #     return nn


# # def transformer_post_process(inputs, pre_attn, out_channel, patch_size=2, patch_height=-1, activation="swish", name=""):
# #     if patch_height == -1:  # V1, [batch * 4, height // 2, width // 2, channel]
# #         patch_hh, patch_ww, channel = inputs.shape[1], inputs.shape[2], inputs.shape[-1]
# #     else:  # V2, [batch, 4, height // 2 * width // 2, channel]
# #         patch_hh, patch_ww, channel = patch_height, inputs.shape[2] // patch_height, inputs.shape[-1]
# #     # print(f"{patch_hh = }, {patch_ww = }, {channel = }, {inputs.shape = }")

# #     # [batch * 4, height // 2, width // 2, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch, height, width, channel]
# #     nn = tf.reshape(inputs, [-1, patch_size * patch_size, patch_hh, patch_ww * channel])  # [batch, h_patch_size * w_patch_size, patch_hh, patch_ww * channel]
# #     nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch, patch_hh, h_patch_size * w_patch_size, patch_ww * channel]
# #     nn = tf.reshape(nn, [-1, patch_size, patch_ww, channel])  # [batch * patch_hh * h_patch_size, w_patch_size, patch_ww, channel]
# #     nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_hh * h_patch_size, patch_ww, w_patch_size, channel]
# #     nn = tf.reshape(nn, [-1, patch_hh * patch_size, patch_ww * patch_size, channel])
# #     # print(f"transformer_post_process before resize: {nn.shape = }")
# #     if pre_attn is not None and (nn.shape[1] != pre_attn.shape[1] or nn.shape[2] != pre_attn.shape[2]):
# #         nn = tf.image.resize(nn, [pre_attn.shape[1], pre_attn.shape[2]], method="bilinear")
# #     # print(f"transformer_post_process after resize: {nn.shape = }")

# #     nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "post_1_")
# #     nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_1_")
# #     if pre_attn is not None:  # V1
# #         nn = tf.concat([pre_attn, nn], axis=-1)
# #         nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=1, padding="SAME", name=name + "post_2_")
# #         nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_2_")
# #     return keras.layers.Activation("linear", name=name + "output")(nn)  # Identity, Just need a name here

# # def se_attention(inputs, name, reduction=4):
# #     b, h, w, c = inputs.shape
# #     sq = keras.layers.GlobalAveragePooling2D(name="SE_pool_"+name)(inputs)
# #     sq = tf.reshape(sq, (-1,c))
# #     fc = keras.layers.Dense(c // reduction, use_bias=False, name="SE_ds1_"+name)(sq)
# #     fc = tf.nn.swish(fc)
# #     fc = keras.layers.Dense(c, use_bias=False, name="SE_ds2_"+name)(fc)
# #     fc = tf.reshape(fc, (-1,1,1,c))

# #     # print( inputs.shape)

# #     return inputs * tf.broadcast_to(fc, (-1,h,w,c))

# # class SELayer(keras.layers.Layer):
# #     def __init__(self, channel, reduction=4, **kwargs):
# #         super(SELayer, self).__init__(**kwargs)
# #         self.channel = channel
# #         self.reduction = reduction
# #         self.pool = keras.layers.GlobalAveragePooling2D()
# #         self.dense1 = keras.layers.Dense(channel // reduction, use_bias=False)
# #         self.dense2 = tf.keras.layers.Dense(channel, use_bias=False)

# #     def get_config(self):
# #         config = super().get_config()
# #         config.update({
# #             "channel": self.channel,
# #             "reduction": self.reduction,
# #         })
# #         return config

# #     def call(self, inputs):
# #         b, h, w, c = inputs.shape
        
# #         sq = self.pool(inputs)
# #         sq = tf.reshape(sq, (-1,c))
# #         fc = self.dense1(sq)
# #         fc = tf.nn.swish(fc)
# #         fc = self.dense2(fc)
# #         fc = tf.reshape(fc, (-1,1,1,c))
# #         print(tf.broadcast_to(fc, (-1,h,w,c)).shape)
# #         return inputs * tf.broadcast_to(fc, (-1,h,w,c))

# # def MobileViT(
# #     num_blocks=[1, 3, 3, 5, 4],
# #     out_channels=[32, 64, 96, 128, 160],
# #     attn_channels=[0, 0, 144, 192, 240],  # Can be a list matching out_channels, or a float number for expansion ratio of out_channels
# #     block_types=["conv", "conv", "transform", "transform", "transform"],
# #     strides=[1, 2, 2, 2, 2],
# #     expand_ratio=4,
# #     stem_width=16,
# #     patch_size=2,
# #     patches_to_batch=True,  # True for V1, False for V2
# #     resize_first=False,  # False for V1, True for V2
# #     use_depthwise=False,  # False for V1, True for V2
# #     use_fusion=True,  # True for V1, False for V2
# #     num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
# #     use_linear_attention=False,  # False for V1, True for V2
# #     use_conv_mlp=False,  # False for V1, True for V2
# #     output_num_features=640,
# #     layer_scale=-1,
# #     input_shape=(256, 256, 3),
# #     num_classes=1000,
# #     activation="swish",
# #     drop_connect_rate=0,
# #     classifier_activation="softmax",
# #     dropout=0,
# #     pretrained=None,
# #     model_name="mobilevit",
# #     kwargs=None,
# # ):
# #     inputs = keras.layers.Input(input_shape)
# #     nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=1, padding="same", name="stem_")
# #     nn = batchnorm_with_activation(nn, activation=activation, name="stem_")

# #     # Save line width
# #     mhsa_mlp_block_common_kwargs = {
# #         "num_heads": 4,
# #         "qkv_bias": True,
# #         "mlp_ratio": 2,
# #         "num_norm_groups": num_norm_groups,
# #         "use_linear_attention": use_linear_attention,
# #         "use_conv_mlp": use_conv_mlp,
# #         "activation": activation,
# #     }

# #     """ stages """
# #     total_blocks = sum(num_blocks)
# #     global_block_id = 0
# #     post_activation = activation if use_fusion else None
# #     for id, (num_block, out_channel, block_type, stride) in enumerate(zip(num_blocks, out_channels, block_types, strides)):
# #         stack_name = "stack{}_".format(id + 1)
# #         is_conv_block = True if block_type[0].lower() == "c" else False
# #         attn_channel = attn_channels[id] if isinstance(attn_channels, (list, tuple)) else make_divisible(attn_channels * out_channel, divisor=8)
# #         for block_id in range(num_block):
# #             name = stack_name + "block{}_".format(block_id + 1)
# #             stride = stride if block_id == 0 else 1
# #             use_shortcut = False if stride != 1 or nn.shape[-1] != out_channel else True
# #             block_drop_rate = drop_connect_rate * global_block_id / total_blocks
# #             global_block_id += 1
# #             if is_conv_block or block_id == 0:  # First transformer block is also a conv block .
# #                 nn = bottle_in_linear_out_block(nn, out_channel, stride, expand_ratio, use_shortcut, block_drop_rate, activation=activation, name=name)
# #             else:
# #                 if block_id == 1:  # pre
# #                     pre_attn = nn if use_fusion else None
# #                     patch_height = -1 if patches_to_batch else int(tf.math.ceil(nn.shape[1] / patch_size))
# #                     nn = transformer_pre_process(nn, attn_channel, patch_size, resize_first, use_depthwise, patches_to_batch, activation=activation, name=name)
# #                 nn = mhsa_mlp_block(nn, attn_channel, layer_scale=layer_scale, **mhsa_mlp_block_common_kwargs, name=name)
# #                 # nn = SELayer(attn_channel)(nn)
# #                 nn = se_attention(nn, name)
# #                 if block_id == num_block - 1:  # post
# #                     nn = group_norm(nn, groups=num_norm_groups, name=name + "post_") if num_norm_groups > 0 else layer_norm(nn, name=name + "post_")
# #                     nn = transformer_post_process(nn, pre_attn, out_channel, patch_size, patch_height, activation=post_activation, name=name)
# #     nn = output_block(nn, output_num_features, activation, num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
# #     model = keras.models.Model(inputs, nn, name=model_name)
# #     add_pre_post_process(model, rescale_mode="raw01")
# #     reload_model_weights(model, PRETRAINED_DICT, "mobilevit", pretrained)
# #     return model


# # def MobileViT_XXS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
# #     num_blocks = [1, 3, 3, 5, 4]
# #     out_channels = [16, 24, 48, 64, 80]
# #     attn_channels = [0, 0, 64, 80, 96]
# #     output_num_features = 320
# #     expand_ratio = 2
# #     return MobileViT(**locals(), model_name="mobilevit_xxs", **kwargs)


# # def MobileViT_XS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
# #     num_blocks = [1, 3, 3, 5, 4]
# #     out_channels = [32, 48, 64, 80, 96]
# #     attn_channels = 1.5
# #     output_num_features = 384
# #     return MobileViT(**locals(), model_name="mobilevit_xs", **kwargs)


# # def MobileViT_S(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
# #     num_blocks = [1, 3, 3, 5, 4]
# #     out_channels = [32, 64, 96, 128, 160]
# #     attn_channels = 1.5
# #     return MobileViT(**locals(), model_name="mobilevit_s", **kwargs)

# if __name__ == "__main__":
#     import numpy as np
#     model = MobileViT_XS()
#     model(np.ones((1,256,256,3)))

import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    ChannelAffine,
    depthwise_conv2d_no_bias,
    drop_block,
    drop_connect_rates_split,
    group_norm,
    layer_norm,
    make_divisible,
    mlp_block,
    multi_head_self_attention,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights
# import tensorflow_models as tfm

class SqueezeExcitation(tf.keras.layers.Layer):
  """Creates a squeeze and excitation layer."""

  def __init__(self,
               in_filters,
               out_filters,
               se_ratio,
               # divisible_by=1,
               # use_3d_input=False,
               # kernel_initializer='VarianceScaling',
               # kernel_regularizer=None,
               # bias_regularizer=None,
               # activation='relu',
               # gating_activation='sigmoid',
               # round_down_protect=True,
               **kwargs):
    """Initializes a squeeze and excitation layer.
    Args:
      in_filters: An `int` number of filters of the input tensor.
      out_filters: An `int` number of filters of the output tensor.
      se_ratio: A `float` or None. If not None, se ratio for the squeeze and
        excitation layer.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      use_3d_input: A `bool` of whether input is 2D or 3D image.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      gating_activation: A `str` name of the activation function for final
        gating function.
      round_down_protect: A `bool` of whether round down more than 10% will be
        allowed.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(SqueezeExcitation, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio
    # self._divisible_by = divisible_by
    # self._round_down_protect = round_down_protect
    # self._use_3d_input = use_3d_input
    # self._activation = activation
    # self._gating_activation = gating_activation
    # self._kernel_initializer = kernel_initializer
    # self._kernel_regularizer = kernel_regularizer
    # self._bias_regularizer = bias_regularizer
    # if tf.keras.backend.image_data_format() == 'channels_last':
    #   if not use_3d_input:
    #     self._spatial_axis = [1, 2]
    #   else:
    #     self._spatial_axis = [1, 2, 3]
    # else:
    #   if not use_3d_input:
    #     self._spatial_axis = [2, 3]
    #   else:
    #     self._spatial_axis = [2, 3, 4]
    # self._activation_fn = tf_utils.get_activation(activation)
    # self._gating_activation_fn = tf_utils.get_activation(gating_activation)

  def build(self, input_shape):
    num_reduced_filters = max(1, int(self._in_filters * self._se_ratio))
    # num_reduced_filters = make_divisible(
    #     max(1, int(self._in_filters * self._se_ratio)),
    #     divisor=self._divisible_by,
    #     round_down_protect=self._round_down_protect)

    self._se_reduce = tf.keras.layers.Conv2D(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        # kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        # kernel_regularizer=self._kernel_regularizer,
        # bias_regularizer=self._bias_regularizer
        )

    self._se_expand = tf.keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        # kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        # kernel_regularizer=self._kernel_regularizer,
        # bias_regularizer=self._bias_regularizer
        )

    super(SqueezeExcitation, self).build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'se_ratio': self._se_ratio,
        # 'divisible_by': self._divisible_by,
        # 'use_3d_input': self._use_3d_input,
        # 'kernel_initializer': self._kernel_initializer,
        # 'kernel_regularizer': self._kernel_regularizer,
        # 'bias_regularizer': self._bias_regularizer,
        # 'activation': self._activation,
        # 'gating_activation': self._gating_activation,
        # 'round_down_protect': self._round_down_protect,
    }
    base_config = super(SqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = tf.nn.swish(self._se_reduce(x))
    x = tf.nn.sigmoid(self._se_expand(x))
    return x * inputs

LAYER_NORM_EPSILON = 1e-5
PRETRAINED_DICT = {
    "mobilevit_xxs": {"imagenet": "f9d1d4f7329726b4bb9069cd631a152e"},
    "mobilevit_xs": {"imagenet": "dcd8591668946aa4ddb0159dfe65cc2b"},
    "mobilevit_s": {"imagenet": "55f1051035ecd14e90ae3df80fd0c7f3"},
    "mobilevit_v2_050": {"imagenet": {256: "a842a40c0f49dc2bbe935493caed061b"}},
    "mobilevit_v2_075": {"imagenet": {256: "8588b3d6bf4aa750766ddc6d01824c67"}},
    "mobilevit_v2_100": {"imagenet": {256: "55d499bbc29f0f6379a4cc6f610e10e8"}},
    "mobilevit_v2_125": {"imagenet": {256: "b8af7b7668774796530f19dd5b6080fb"}},
    "mobilevit_v2_150": {
        "imagenet": {256: "065e7a07f7e2d0d74a33913195df9044"},
        "imagenet22k": {256: "cf3c4ec278154ece62e8967faa5c0391", 384: "cdcfaebb573f8cd1f41044ac0e958204"},
    },
    "mobilevit_v2_175": {
        "imagenet": {256: "627719428c6cb35f071a7ea69a6961c4"},
        "imagenet22k": {256: "b849708a6b2c1f115b8b8c366e1d1a19", 384: "d3feef5108b6195d1c5525fb185bf720"},
    },
    "mobilevit_v2_200": {
        "imagenet": {256: "1fe59d8bb2662761084d1c04259a778d"},
        "imagenet22k": {256: "931f0be1761bcf8443359ec1661bb6a7", 384: "1dc6cdafb187611e5a4819272d64fba7"},
    },
}


def bottle_in_linear_out_block(inputs, out_channel, strides=1, expand_ratio=4, use_shortcut=False, drop_rate=0, activation="swish", name=""):
    hidden_dim = int(inputs.shape[-1] * expand_ratio)
    deep = conv2d_no_bias(inputs, hidden_dim, kernel_size=1, strides=1, name=name + "deep_1_")
    deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_1_")
    deep = depthwise_conv2d_no_bias(deep, kernel_size=3, strides=strides, padding="SAME", name=name + "deep_2_")
    deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_2_")
    deep = conv2d_no_bias(deep, out_channel, kernel_size=1, strides=1, name=name + "deep_3_")
    deep = batchnorm_with_activation(deep, activation=None, name=name + "deep_3_")
    deep = drop_block(deep, drop_rate=drop_rate, name=name + "deep_")

    out = keras.layers.Add()([inputs, deep]) if use_shortcut else deep
    return out


def linear_self_attention(inputs, qkv_bias=False, out_bias=False, attn_axis=2, attn_dropout=0, name=None):
    input_channel = inputs.shape[-1]
    qkv = conv2d_no_bias(inputs, 1 + input_channel * 2, kernel_size=1, use_bias=qkv_bias, name=name and name + "qkv_")
    query, key, value = tf.split(qkv, [1, input_channel, input_channel], axis=-1)
    context_score = keras.layers.Softmax(axis=attn_axis, name=name and name + "attention_scores")(query)  # on patch_hh * patch_ww dimension
    context_score = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(context_score) if attn_dropout > 0 else context_score
    # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {context_score.shape = }")

    context_vector = keras.layers.Multiply()([key, context_score])  # [batch, height, width, input_channel]
    context_vector = tf.reduce_sum(context_vector, keepdims=True, axis=attn_axis)  # on patch_hh * patch_ww dimension

    out = tf.nn.relu(value) * context_vector
    out = conv2d_no_bias(out, input_channel, kernel_size=1, use_bias=out_bias, name=name and name + "output")
    return out


def mhsa_mlp_block(
    inputs,
    out_channel,
    num_heads=4,
    qkv_bias=True,
    mlp_ratio=4,
    num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
    use_linear_attention=False,  # False for V1, True for V2
    use_conv_mlp=False,  # False for V1, True for V2
    mlp_drop_rate=0,
    attn_drop_rate=0,
    drop_rate=0,
    layer_scale=-1,
    activation="gelu",
    name=None,
):
    attn = group_norm(inputs, groups=num_norm_groups, name=name + "attn_") if num_norm_groups > 0 else layer_norm(inputs, name=name + "attn_")
    if use_linear_attention:  # V2
        if num_norm_groups > 0:
            attn = keras.layers.Reshape(attn.shape[1:])(attn)  # Or will throw error when converting tflite, if GroupNorm is followed by Conv2D
        attn = linear_self_attention(attn, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
    else:  # V1
        attn = multi_head_self_attention(attn, num_heads, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name and name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name and name + "attn_")
    attn_out = keras.layers.Add(name=name and name + "attn_out")([inputs, attn])

    mlp = group_norm(attn_out, groups=num_norm_groups, name=name + "mlp_") if num_norm_groups > 0 else layer_norm(attn_out, name=name + "mlp_")
    if use_conv_mlp and num_norm_groups > 0:  # V2
        mlp = keras.layers.Reshape(mlp.shape[1:])(mlp)  # Or will throw error when converting tflite, if GroupNorm is followed by Conv2D
    mlp = mlp_block(mlp, int(out_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=use_conv_mlp, activation=activation, name=name and name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name and name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name and name + "mlp_")
    return keras.layers.Add(name=name and name + "out")([attn_out, mlp])


def transformer_pre_process(inputs, out_channel, patch_size=2, resize_first=False, use_depthwise=False, patches_to_batch=True, activation="swish", name=""):
    nn = inputs

    if resize_first:  # V2
        patch_hh, patch_ww = int(tf.math.ceil(nn.shape[1] / patch_size)), int(tf.math.ceil(nn.shape[2] / patch_size))
        # print(f"transformer_pre_process before resize: {nn.shape = }")
        if patch_hh * patch_size != nn.shape[1] or patch_ww * patch_size != nn.shape[2]:
            nn = tf.image.resize(nn, [patch_hh * patch_size, patch_ww * patch_size], method="bilinear")

    if use_depthwise:  # V2
        nn = depthwise_conv2d_no_bias(nn, kernel_size=3, strides=1, padding="SAME", name=name + "pre_1_")
    else:  # V1
        nn = conv2d_no_bias(nn, nn.shape[-1], kernel_size=3, strides=1, padding="SAME", name=name + "pre_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "pre_1_")
    nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "pre_2_")

    if not resize_first:  # V1
        patch_hh, patch_ww = int(tf.math.ceil(nn.shape[1] / patch_size)), int(tf.math.ceil(nn.shape[2] / patch_size))
        # print(f"transformer_pre_process before resize: {nn.shape = }")
        if patch_hh * patch_size != nn.shape[1] or patch_ww * patch_size != nn.shape[2]:
            nn = tf.image.resize(nn, [patch_hh * patch_size, patch_ww * patch_size], method="bilinear")

    # Extract patchs, limit transpose permute length <= 4
    # [batch, height, width, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch * 4, height // 2, width // 2, channel]
    # print(f"transformer_pre_process after resize: {nn.shape = }")
    nn = tf.reshape(nn, [-1, patch_ww, patch_size, out_channel])  # [batch * patch_hh * h_patch_size, patch_ww, w_patch_size, channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_hh * h_patch_size, w_patch_size, patch_ww, channel]
    nn = tf.reshape(nn, [-1, patch_hh, patch_size * patch_size, patch_ww * out_channel])  # [batch, patch_hh, h_patch_size * w_patch_size, patch_ww * channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch, h_patch_size * w_patch_size, patch_hh, patch_ww * channel]
    extract_shape = [-1, patch_hh, patch_ww, out_channel] if patches_to_batch else [-1, patch_size * patch_size, patch_hh * patch_ww, out_channel]
    nn = tf.reshape(nn, extract_shape)

    return nn


def transformer_post_process(inputs, pre_attn, out_channel, patch_size=2, patch_height=-1, activation="swish", name=""):
    if patch_height == -1:  # V1, [batch * 4, height // 2, width // 2, channel]
        patch_hh, patch_ww, channel = inputs.shape[1], inputs.shape[2], inputs.shape[-1]
    else:  # V2, [batch, 4, height // 2 * width // 2, channel]
        patch_hh, patch_ww, channel = patch_height, inputs.shape[2] // patch_height, inputs.shape[-1]
    # print(f"{patch_hh = }, {patch_ww = }, {channel = }, {inputs.shape = }")

    # [batch * 4, height // 2, width // 2, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch, height, width, channel]
    nn = tf.reshape(inputs, [-1, patch_size * patch_size, patch_hh, patch_ww * channel])  # [batch, h_patch_size * w_patch_size, patch_hh, patch_ww * channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch, patch_hh, h_patch_size * w_patch_size, patch_ww * channel]
    nn = tf.reshape(nn, [-1, patch_size, patch_ww, channel])  # [batch * patch_hh * h_patch_size, w_patch_size, patch_ww, channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_hh * h_patch_size, patch_ww, w_patch_size, channel]
    nn = tf.reshape(nn, [-1, patch_hh * patch_size, patch_ww * patch_size, channel])
    # print(f"transformer_post_process before resize: {nn.shape = }")
    if pre_attn is not None and (nn.shape[1] != pre_attn.shape[1] or nn.shape[2] != pre_attn.shape[2]):
        nn = tf.image.resize(nn, [pre_attn.shape[1], pre_attn.shape[2]], method="bilinear")
    # print(f"transformer_post_process after resize: {nn.shape = }")

    nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "post_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_1_")
    if pre_attn is not None:  # V1
        nn = tf.concat([pre_attn, nn], axis=-1)
        nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=1, padding="SAME", name=name + "post_2_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_2_")
    return keras.layers.Activation("linear", name=name + "output")(nn)  # Identity, Just need a name here


def MobileViT(
    num_blocks=[1, 3, 3, 5, 4],
    out_channels=[32, 64, 96, 128, 160],
    attn_channels=[0, 0, 144, 192, 240],  # Can be a list matching out_channels, or a float number for expansion ratio of out_channels
    block_types=["conv", "conv", "transform", "transform", "transform"],
    strides=[1, 2, 2, 2, 2],
    expand_ratio=4,
    stem_width=16,
    patch_size=2,
    patches_to_batch=True,  # True for V1, False for V2
    resize_first=False,  # False for V1, True for V2
    use_depthwise=False,  # False for V1, True for V2
    use_fusion=True,  # True for V1, False for V2
    num_norm_groups=-1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
    use_linear_attention=False,  # False for V1, True for V2
    use_conv_mlp=False,  # False for V1, True for V2
    output_num_features=640,
    layer_scale=-1,
    input_shape=(256, 256, 3),
    num_classes=1000,
    activation="swish",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="mobilevit",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=2, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")

    # Save line width
    mhsa_mlp_block_common_kwargs = {
        "num_heads": 4,
        "qkv_bias": True,
        "mlp_ratio": 2,
        "num_norm_groups": num_norm_groups,
        "use_linear_attention": use_linear_attention,
        "use_conv_mlp": use_conv_mlp,
        "activation": activation,
    }

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    post_activation = activation if use_fusion else None
    for id, (num_block, out_channel, block_type, stride) in enumerate(zip(num_blocks, out_channels, block_types, strides)):
        stack_name = "stack{}_".format(id + 1)
        is_conv_block = True if block_type[0].lower() == "c" else False
        attn_channel = attn_channels[id] if isinstance(attn_channels, (list, tuple)) else make_divisible(attn_channels * out_channel, divisor=8)
        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            stride = stride if block_id == 0 else 1
            use_shortcut = False if stride != 1 or nn.shape[-1] != out_channel else True
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            if is_conv_block or block_id == 0:  # First transformer block is also a conv block .
                nn = bottle_in_linear_out_block(nn, out_channel, stride, expand_ratio, use_shortcut, block_drop_rate, activation=activation, name=name)
            else:
                if block_id == 1:  # pre
                    pre_attn = nn if use_fusion else None
                    patch_height = -1 if patches_to_batch else int(tf.math.ceil(nn.shape[1] / patch_size))
                    nn = transformer_pre_process(nn, attn_channel, patch_size, resize_first, use_depthwise, patches_to_batch, activation=activation, name=name)
                nn = mhsa_mlp_block(nn, attn_channel, layer_scale=layer_scale, **mhsa_mlp_block_common_kwargs, name=name)
                nn = SqueezeExcitation(attn_channel, attn_channel, 0.25)(nn)
                if block_id == num_block - 1:  # post
                    nn = group_norm(nn, groups=num_norm_groups, name=name + "post_") if num_norm_groups > 0 else layer_norm(nn, name=name + "post_")
                    nn = transformer_post_process(nn, pre_attn, out_channel, patch_size, patch_height, activation=post_activation, name=name)

    nn = output_block(nn, output_num_features, activation, num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="raw01")
    reload_model_weights(model, PRETRAINED_DICT, "mobilevit", pretrained)
    return model


def MobileViT_XXS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [16, 24, 48, 64, 80]
    attn_channels = [0, 0, 64, 80, 96]
    output_num_features = 320
    expand_ratio = 2
    return MobileViT(**locals(), model_name="mobilevit_xxs", **kwargs)


def MobileViT_XS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [32, 48, 64, 80, 96]
    attn_channels = 1.5
    output_num_features = 384
    return MobileViT(**locals(), model_name="mobilevit_xs", **kwargs)


def MobileViT_S(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [32, 64, 96, 128, 160]
    attn_channels = 1.5
    return MobileViT(**locals(), model_name="mobilevit_s", **kwargs)