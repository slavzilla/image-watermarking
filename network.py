import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, ReLU, LeakyReLU, Flatten, Dense, Concatenate, Conv2DTranspose, Add, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid, relu
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import BinaryCrossentropy
from utils import dft, savitzky_golay, Weights, get_watermark
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
x_train, x_test = x_train / 255.0, x_test / 255.0

BATCH_SIZE = 64
BUFFER_SIZE = 128
IMAGE_SHAPE = (32, 32, 3)
MESSAGE_SHAPE = (4, 4, 3)
EPOCH = 1
EPOCH_LEN = 50000 // BATCH_SIZE
NUM_EPOCHS = 40

weights = Weights()

def get_embedder():
    inputImage = Input(shape = IMAGE_SHAPE)
    inputMessage = Input(shape = MESSAGE_SHAPE)
    
    conv_0 = Conv2D(filters=8, kernel_size=5, strides=1, padding='same')(inputImage) 
    conv_0 = ReLU()(conv_0)

    conv_1 = Conv2D(filters=16, kernel_size=5, strides=1, padding='same')(conv_0) 
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = ReLU()(conv_1)

    conv_2 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(conv_1) 
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = ReLU()(conv_2)

    conv_3 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(conv_2) # 16 x 16
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = ReLU()(conv_3)

    conv_4 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = ReLU()(conv_4)

    conv_5 = Conv2D(filters=256, kernel_size=1, strides=2, padding='same')(conv_4) # None x 4 x 4 x 256
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = ReLU()(conv_5)

    bottleneck = Concatenate()([conv_5, inputMessage])

    conv_6 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation=relu)(bottleneck) # None x 4 x 4 x 256

    upsampling_1 = Conv2DTranspose(filters=128, kernel_size=1, strides=2, padding='same')(conv_6) # bilo 5 pa sad 1
    upsampling_1 = Concatenate()([upsampling_1, conv_4])
 
    conv_7 = BatchNormalization()(upsampling_1)
    conv_7 = ReLU()(conv_7)
    conv_7 = Dropout(rate=0.5)(conv_7)

    upsampling_2 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(conv_7)
    upsampling_2 = Concatenate()([upsampling_2, conv_3])

    conv_8 = BatchNormalization()(upsampling_2)
    conv_8 = ReLU()(conv_8)
    conv_8 = Dropout(rate=0.5)(conv_8)

    upsampling_3 = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(conv_8)
    upsampling_3 = Concatenate()([upsampling_3, conv_2])

    conv_9 = BatchNormalization()(upsampling_3)
    conv_9 = ReLU()(conv_9)
    conv_9 = Dropout(rate=0.5)(conv_9)

    upsampling_4 = Conv2DTranspose(filters=16, kernel_size=5, strides=1, padding='same')(conv_9)
    upsampling_4 = Concatenate()([upsampling_4, conv_1])

    conv_10 = BatchNormalization()(upsampling_4)
    conv_10 = ReLU()(conv_10)

    upsampling_5 = Conv2DTranspose(filters=8, kernel_size=5, strides=1, padding='same')(conv_10)
    upsampling_5 = Concatenate()([upsampling_5, conv_0])

    conv_11 = BatchNormalization()(upsampling_5)
    conv_11 = ReLU()(conv_11)


    output = Conv2D(filters=IMAGE_SHAPE[-1], kernel_size=5, strides=1, padding='same', activation=sigmoid)(conv_11) 
    return Model(inputs = [inputImage, inputMessage], outputs = output)

def get_detector():

    detectorInput = Input(shape=IMAGE_SHAPE)

    dconv_1 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(detectorInput)
    dconv_1 = BatchNormalization()(dconv_1)
    dconv_1 = LeakyReLU(alpha=0.2)(dconv_1)

    dconv_2 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(dconv_1)
    dconv_2 = BatchNormalization()(dconv_2)
    dconv_2 = LeakyReLU(alpha=0.2)(dconv_2)

    dconv_3 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dconv_2)
    dconv_3 = BatchNormalization()(dconv_3)
    dconv_3 = LeakyReLU(alpha=0.2)(dconv_3)

    dconv_4 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dconv_3)
    dconv_4 = BatchNormalization()(dconv_4)
    dconv_4 = LeakyReLU(alpha=0.2)(dconv_4)

    dconv_5 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dconv_4)
    dconv_5 = BatchNormalization()(dconv_5)
    dconv_5 = LeakyReLU(alpha=0.2)(dconv_5)

    dconv_6 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same')(dconv_5)
    dconv_6 = BatchNormalization()(dconv_6) 
    dconv_6 = LeakyReLU(alpha=0.2)(dconv_6)

    flatten = Flatten()(dconv_6)

    dense = Dense(units=MESSAGE_SHAPE[-1], activation=sigmoid)(flatten)
    return Model(inputs=detectorInput, outputs=dense)

def get_model():
    encoder = get_embedder()
    decoder = get_detector()
    inp1 = Input(IMAGE_SHAPE)
    inp2 = Input(MESSAGE_SHAPE)
    e = encoder([inp1, inp2])
    d = decoder(e)
    return Model(inputs=[inp1, inp2], outputs=[e,d])

def create_dataset(input, num_epochs=NUM_EPOCHS, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((input))
    dataset = dataset.repeat(count=num_epochs)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

def generator(iterator):
  try:
    while True:
      yield [next(iterator), get_watermark(BATCH_SIZE, MESSAGE_SHAPE[-1])]
  except (RuntimeError, StopIteration):
    return

def embedderLossFunction(y_true, y_pred):
    im1 = tf.image.convert_image_dtype(y_true, tf.float32)
    im2 = tf.image.convert_image_dtype(y_pred, tf.float32)
    return tf.reduce_mean(1.0 - tf.image.ssim(im1, im2, 1.0))

def detectorLossFunction(y_true, y_pred):
    y_augmented = tf.squeeze(tf.slice(y_true, [0, 0, 0, 0], [BATCH_SIZE, 1, 1, MESSAGE_SHAPE[-1]]))
    #print(y_augmented[0], y_pred[0])
    return BinaryCrossentropy()(y_augmented, y_pred)

def customLoss(encoderLoss, decoderLoss):
  encoder_loss_weight, decoder_loss_weight = weights.get_weights(EPOCH)
  return Add()([Multiply()([encoder_loss_weight, encoderLoss]), Multiply()([decoder_loss_weight, decoderLoss])])

def compute_loss(model, input):
  e, d = model(input)
  inputSignal, inputMessage = input
  e_loss = embedderLossFunction(inputSignal, e)
  d_loss = detectorLossFunction(inputMessage, d)
  return customLoss(e_loss, d_loss), e_loss, d_loss

def train_step(model, input, optimizer):
  with tf.GradientTape() as tape:
    loss, e_loss, d_loss = compute_loss(model, input)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, e_loss, d_loss

def test_step(model, input):
    input_signal, input_message = input
    embedder_output, detector_output = model(input, training=False)
    output_messages = tf.where(tf.greater_equal(detector_output, 0.5), 1, 0)
    input_message = tf.cast(tf.squeeze(tf.slice(input_message, [0, 0, 0, 0], [BATCH_SIZE, 1, 1,  MESSAGE_SHAPE[-1]])), tf.int32)
    mask = tf.where(tf.equal(output_messages, input_message), 1, 0).numpy()
    count = input_signal.shape[0]
    im1 = tf.image.convert_image_dtype(input_signal, tf.float32)
    im2 = tf.image.convert_image_dtype(embedder_output, tf.float32)
    ssim = tf.image.ssim(im1, im2, 1.0).numpy()
    return np.sum(mask), np.sum(ssim), count

def test(model, it):
    total_ssim = 0
    count = 0
    total_acc = 0
    step = 1
    for batch in generator(it):
        batch_acc, batch_ssim, batch_count = test_step(model, batch)
        total_ssim += batch_ssim
        total_acc += batch_acc
        count += batch_count
        step += 1
    return total_acc/(count*MESSAGE_SHAPE[-1]), total_ssim / count

def restore_model():
    model = get_model()
    dataset = create_dataset(x_train)
    it = iter(dataset)
    optimizer= Nadam(learning_rate=1*1e-5)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, it=it)
    manager = tf.train.CheckpointManager(ckpt, "image_watermarking", max_to_keep=20)
    manager.restore_or_initialize()
    test_dataset = create_dataset(x_test, 1)
    test_it = iter(test_dataset)
    return model, test_it

"""

dataset = create_dataset(x_train)
it = iter(dataset)
test_dataset = create_dataset(x_test, 1)
test_it = iter(test_dataset)

model = get_model()
optimizer= Nadam(learning_rate=1*1e-5)

ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer,  it=it)
manager = tf.train.CheckpointManager(ckpt, "image_watermarking", max_to_keep=20)
manager.restore_or_initialize()

e_losses = []
d_losses = []
accuracies = []
ssims = []
step = 0
for batch in generator(it):
    loss, e_loss, d_loss, = train_step(model, batch, optimizer)
    step += 1
    EPOCH_OLD = EPOCH
    EPOCH = int(step / EPOCH_LEN) + 1
    if step % 50 == 0:
        print("Current loss is:", loss.numpy(), ", embedder loss is:", e_loss.numpy(), "while detector loss is:", d_loss.numpy())
        e_losses.append(e_loss.numpy())
        d_losses.append(d_loss.numpy())
        weights.print(EPOCH)
    if EPOCH % 2 == 0 and EPOCH_OLD != EPOCH:
        print("====================VALIDATING====================")
        test_it = iter(test_dataset)
        acc, ssim = test(model, test_it)
        accuracies.append(acc)
        ssims.append(ssim)
        print("Current accuracy is:", acc, "while current ssim is:", ssim)
        print("====================VALIDATION DONE====================")
        if(acc > 0.99 and ssim > 0.97):
            manager.save()

np.save("e_losses.npy", np.array(e_losses))
np.save("d_losses.npy", np.array(d_losses))
np.save("accuracies.npy", np.array(accuracies))
np.save("ssims", np.array(ssims))

"""