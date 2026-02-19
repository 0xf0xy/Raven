import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
def masked_loss(y_true, y_pred):
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, True)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    correct_predictions = tf.multiply(loss, mask)

    return tf.reduce_mean(correct_predictions)


@keras.saving.register_keras_serializable()
def masked_accuracy(y_true, y_pred):
    accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    correct_predictions = tf.multiply(accuracy, mask)

    return tf.reduce_sum(correct_predictions) / tf.reduce_sum(mask)
