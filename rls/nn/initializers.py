import tensorflow as tf

# initKernelAndBias = {
#     'kernel_initializer': tf.random_normal_initializer(0.0, .1),
#     'bias_initializer': tf.constant_initializer(0.1)    # 2.x 不需要指定dtype
# }

initKernelAndBias = {
    'kernel_initializer': 'he_normal',
    'bias_initializer': 'he_normal'
}
