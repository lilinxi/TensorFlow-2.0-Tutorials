import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

# (60000, 28, 28) -> (1000, 784)
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# Define a simple sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# You can use a trained model without having to retrain it,
# or pick-up training where you left off—in case the training process was interrupted.
# The tf.keras.callbacks.ModelCheckpoint callback allows to continually save the model
# both during and at the end of training.

# Save Checkpoints
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_dir)  # training_1

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

# Load Checkpoints

# Create a basic model instance
model = create_model()
# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
# Loads the weights
model.load_weights(checkpoint_path)
# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Options

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)
# Create a new model instance
model = create_model()
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))
# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=50,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
# Create a new model instance
model = create_model()
# Load the previously saved weights
model.load_weights(latest)
# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Manually save weights

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')
# Create a new model instance
model = create_model()
# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')
# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Save the entire model
# This technique saves everything:
#   - The weight values
#   - The model's configuration(architecture)
#   - The optimizer configuration

# 1. HDF5 format
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model shuold be saved to HDF5.
model.save('my_model.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

# 2. SavedModel format
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
model.save('saved_model/my_model')

new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

# If you are using the SavedModel format, you can skip this section.
# The key difference between HDF5 and SavedModel is that HDF5 uses object configs to save the model architecture,
# while SavedModel saves the execution graph.
# Thus, SavedModels are able to save custom objects like
# subclassed models and custom layers without requiring the orginal code.
