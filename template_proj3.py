import math

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

images = np.load('./images.npy')
images = np.reshape(images, (len(images), 28*28))

labels = np.load('./labels.npy')

seed = 42
np.random.seed(seed)

print(images.shape)
print(labels.shape)

labels = tf.keras.utils.to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.40, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.375, random_state=42)
print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)

# Model Template
model = Sequential()  # declare model
model.add(Dense(1024, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))

model.add(Dense(512, kernel_initializer='random_uniform'))
model.add(Activation('relu'))

model.add(Dense(256, kernel_initializer='random_uniform'))
model.add(Activation('relu'))

model.add(Dense(128, kernel_initializer='random_uniform'))
model.add(Activation('relu'))

model.add(Dense(32, kernel_initializer='random_uniform'))
model.add(Activation('relu'))

# Softmax is often used as the activation for the last layer of a classification network
# because the result could be interpreted as a probability distribution.
model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))

print(model.summary())
# Compile Model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data=(x_validation, y_validation),
                    epochs=90,
                    batch_size=256)

# Report Results

print(history.history)
plt.plot(range(len(history.history.get("accuracy"))), history.history.get("accuracy"), label="accuracy")
plt.plot(range(len(history.history.get("val_accuracy"))), history.history.get("val_accuracy"), label="val_accuracy")
plt.legend(loc="lower right")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("accuracy.png")

plt.clf()

plt.plot(range(len(history.history.get("loss"))), history.history.get("loss"), label="loss")
plt.plot(range(len(history.history.get("val_loss"))), history.history.get("val_loss"), label="val_loss")
plt.legend(loc="upper right")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("loss.png")

plt.clf()

prediction = model.predict(x_test)

scores = model.evaluate(x_test, y_test, verbose=0)
print("\n")
print(scores)
accuracy = scores[1] * 100
error = 100 - scores[1] * 100
print("Error: %.2f%%" % error)
print("Accuracy: %.2f%%" % accuracy)

projection, actual = [], []

# Iterate through predictions, determining which value recieved the highest prediction and
#   marking that value in a list (denoting our prediction)
for p in prediction:

    # Reset max value, index, and max index each iteration
    maximum = float(0)
    index, max_index = 0, -1

    # Iterate through each set of preductions determining the highest predicted value
    for n in p:
        if float(n) > maximum:
            maximum = float(n)
            max_index = index
        index += 1

    # Mark our projection for this set in our list
    projection.append(max_index)

# Convert test set (actuals) back to standard numerical format
for t in y_test:
    actual.append(np.argmax(t))


# Generate Confusion Matrix
y_actual = pd.Series(actual, name='Actual')
y_predict = pd.Series(projection, name='Predicted')
confusion_matrix = pd.crosstab(y_actual, y_predict)

# Generate normalized confusion matrix
norm_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)

# Generate full confusion matrix with totals
full_confusion_matrix = pd.crosstab(y_actual, y_predict, rownames=['True Label'], colnames=['Predicted Label'], margins=True)

# print(norm_confusion_matrix)
print(full_confusion_matrix)

cmap = mpl.cm.get_cmap('Oranges')
plt.matshow(confusion_matrix, cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(confusion_matrix.columns))
plt.xticks(tick_marks, confusion_matrix.columns, rotation=45)
plt.yticks(tick_marks, confusion_matrix.index)

plt.ylabel(confusion_matrix.index.name)
plt.xlabel(confusion_matrix.columns.name)

plt.show()

# print("Plotting incorrect predictions...")
#
# reshape_image = np.reshape(x_test, (len(x_test), round(math.sqrt(28*28)), round(math.sqrt(28*28))))
#
# count = 0
#
# for i in range(len(prediction)):
#     if np.argmax(prediction[i]) != np.argmax(y_test[i]):
#         count += 1
#
# fig = plt.figure(figsize=(round(math.sqrt(count)), round(math.sqrt(count))))
# fig.suptitle("Incorrect predictions and their corresponding images", fontsize=20)
#
# i = 0
# j = 1
# while i < len(reshape_image) and j < round(math.sqrt(count)) * round(math.sqrt(count)):
#     if np.argmax(prediction[i]) != np.argmax(y_test[i]):
#         fig.add_subplot(round(math.sqrt(count)), round(math.sqrt(count)), j)
#         j += 1
#         ax = plt.gca()
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)
#         plt.imshow(reshape_image[i], cmap='gray')
#         plt.title(np.argmax(prediction[i]))
#     i += 1
#
# plt.tight_layout()
#
# plt.savefig("incorrect-predictions.png")
