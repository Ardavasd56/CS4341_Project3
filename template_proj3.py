import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
import seaborn as sns

images = np.load("./images.npy")
images = np.reshape(images, (len(images), 28 * 28))

labels = np.load("./labels.npy")

seed = 42
np.random.seed(seed)

print(images.shape)
print(labels.shape)

labels = tf.keras.utils.to_categorical(labels)

x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.40, random_state=seed, stratify=labels
)
x_test, x_validation, y_test, y_validation = train_test_split(
    x_test, y_test, test_size=0.375, random_state=seed, stratify=y_test
)
print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)

# Model Template
# model = Sequential()  # declare model
# model.add(
#     Dense(
#         784,
#         input_shape=(28 * 28,),
#         kernel_initializer="random_uniform",
#         activation="relu",
#     )
# )  # first layer
#
# model.add(Dense(392, kernel_initializer="random_uniform", activation="relu"))
#
# model.add(Dense(196, kernel_initializer="random_uniform", activation="relu"))
#
# # model.add(Dense(98, kernel_initializer="random_uniform", activation="relu"))
#
# model.add(Dense(98, kernel_initializer="glorot_uniform", activation="tanh"))
#
# # Softmax is often used as the activation for the last layer of a classification network
# # because the result could be interpreted as a probability distribution.
# model.add(Dense(10, kernel_initializer="he_normal", activation="softmax"))  # last layer
#
# print(model.summary())
# # Compile Model
# model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
model = load_model("best_trained_model.h5")

# Train Model
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=100,
    batch_size=128,
)

# model.save("best_trained_model.h5")

# Training Performance Plot
print(history.history)
plt.plot(
    range(len(history.history.get("accuracy"))),
    history.history.get("accuracy"),
    label="train_accuracy",
)
plt.plot(
    range(len(history.history.get("val_accuracy"))),
    history.history.get("val_accuracy"),
    label="validation_accuracy",
)
plt.legend(loc="lower right")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("accuracy.png")

plt.clf()

plt.plot(
    range(len(history.history.get("loss"))), history.history.get("loss"), label="train_loss"
)
plt.plot(
    range(len(history.history.get("val_loss"))),
    history.history.get("val_loss"),
    label="validation_loss",
)
plt.legend(loc="upper right")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("loss.png")

plt.clf()

# Model Prediction
prediction = model.predict(x_test)

# Model Evaluation
scores = model.evaluate(x_test, y_test, verbose=0)
print("\n")
print(scores)
accuracy = scores[1] * 100
error = 100 - scores[1] * 100
print("Accuracy: %.2f%%" % accuracy)
print("Error: %.2f%%" % error)

# Confustion Matrix
projection, actual = [], []

for p in prediction:

    maximum = float(0)
    index, max_index = 0, -1

    for n in p:
        if float(n) > maximum:
            maximum = float(n)
            max_index = index
        index += 1

    projection.append(max_index)

for t in y_test:
    actual.append(np.argmax(t))

y_actual = pd.Series(actual, name="Actual")
y_predict = pd.Series(projection, name="Predicted")
confusion_matrix = pd.crosstab(y_actual, y_predict)

full_confusion_matrix = pd.crosstab(
    y_actual,
    y_predict,
    rownames=["True Label"],
    colnames=["Predicted Label"],
    margins=True,
)

print(full_confusion_matrix)

cmap = mpl.cm.get_cmap("GnBu")
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