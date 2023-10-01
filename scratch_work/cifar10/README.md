# CIFAR-10 experimentation

With the MNIST dataset, peak performance was obtained with a 2 layer convolutional network with a dropout layer. There was no difference in performance caused by the addition of pooling layers. And so a 2 layer convolutional network with no pooling layers will be the starting point for experimentation with cifar-10.

# Batch Size effect

Not really observed with MNIST, but with cifar-10 I'm seeing a large difference caused by changing the batch size with no other changes.

Batch Size / Accuracy (no other changes, 2 layers+dropout)
64 / 72.29
128 / 68.99
256 / 67.90
512 / 66.45
1024 / 64.17
2048 / 57.11

ChatGPT said that 3 ways to increase performance accomodating for increases in batch size are to increase the learning rate, strengthening regularization techniques, and early stopping.

By changing Adam's default learning rate of .001 to .003, we saw an increase in accuracy on the test set from 57.11 to 62.73. This was peak as increasing or decreasing from there resulted in worse performance on the test set.

Dropout of .5 is about peak. Going further did not improve performance.

Adding a second dropout layer after the first dropped performance to 59.30

Early stopping gave the best results. Early stopping alone brought the accuracy up to 68.21 with batch size 2048.
Early stopping + increasing the learning rate in adam from .001 to .003 resulted in 70.89.
By increasing the 'patience' of the early stopping from 3 to 5, we got it up to 72.54 after 33 epochs. This outperformed using a batch size of 64 and was also faster.
So to address the batch size effect observed when working with a batch_size for 2048, the best mitigation we found was an increased learning rate combined with early stopping. We did not notice a profound change by strengthening regularization.

# Wider layers
Moving from 32/64 to 64/128 reduced accuracy to 71.67
32/128 => 69.90
128/256 => 72.74

Not seeing any large increase in performance with wider networks. 

# Deeper network
32/64/128 => 74.42
32/64/128/256 => 74.50

# Over-pooling
When adding a fifth and sixth layer, got this error:
Negative dimension size caused by subtracting 2 from 1 for '{{node max_pooling2d_5/MaxPool}} = MaxPool[T=DT_FLOAT, data_format="NHWC", explicit_paddings=[], ksize=[1, 2, 2, 1], padding="VALID", strides=[1, 2, 2, 1]](Placeholder)' with input shapes: [?,1,1,1024].

Call arguments received by layer "max_pooling2d_5" (type MaxPooling2D):
  • inputs=tf.Tensor(shape=(None, 1, 1, 1024), dtype=float32)


English translation - spacial dimensions of my feature maps had been reduced too far by the pooling layers.

1. 32 -> 16
2. 16 -> 8
3. 8 -> 4
4. 4 -> 2
5. 2 -> 1

Trying a 6th time causes the error.

To get around this, I changed the architecture to have a pooling layer after every 2 convolution layers. At 8 convolution layers, accuracy of the network plummeted to 10%. Going deeper is not the answer.


# Transfer Learning
VGG-16 trained on cifar-10 images resized to 112 by 112 scored exactly 10%.
Resnet-50 trained on cifar-10 images resized to 128 x 128 scored 50%

Caused me to question whether the input size needs to be multiples of the original, so retried VGG-16 with images resized to 128x128. Still 10%

First pass training from scratch with resnet 50 got 49.1%.
First pass fine tuning got 46.24, but I used larger batch size of 128. Trying again with batch size of 64 to match what i used with scratch.
47.98 on second pass. Still strangely bad and worse than training from scratch. Going to unfreeze and increase the learning rate.

Much better results unfreezing the network and using adam's default learning rate, but overfitted drastically in the last epoch as evidenced by discrepancy between training and validation accuracy. 

.
782/782 [==============================] - 45s 41ms/step - loss: 1.0880 - accuracy: 0.6559 - val_loss: 2.8421 - val_accuracy: 0.2022
Epoch 2/10
782/782 [==============================] - 30s 38ms/step - loss: 0.7771 - accuracy: 0.7377 - val_loss: 1.7202 - val_accuracy: 0.4500
Epoch 3/10
782/782 [==============================] - 30s 38ms/step - loss: 0.5763 - accuracy: 0.8018 - val_loss: 0.8779 - val_accuracy: 0.7149
Epoch 4/10
782/782 [==============================] - 30s 38ms/step - loss: 0.4520 - accuracy: 0.8421 - val_loss: 1.1734 - val_accuracy: 0.6391
Epoch 5/10
782/782 [==============================] - 30s 39ms/step - loss: 0.5239 - accuracy: 0.8223 - val_loss: 1.5588 - val_accuracy: 0.5238
Epoch 6/10
782/782 [==============================] - 30s 39ms/step - loss: 0.4536 - accuracy: 0.8435 - val_loss: 0.8822 - val_accuracy: 0.7296
Epoch 7/10
782/782 [==============================] - 30s 39ms/step - loss: 0.4732 - accuracy: 0.8392 - val_loss: 0.8317 - val_accuracy: 0.7241
Epoch 8/10
782/782 [==============================] - 30s 39ms/step - loss: 0.2346 - accuracy: 0.9186 - val_loss: 0.8037 - val_accuracy: 0.7419
Epoch 9/10
782/782 [==============================] - 30s 39ms/step - loss: 0.1402 - accuracy: 0.9542 - val_loss: 0.6691 - val_accuracy: 0.8260
Epoch 10/10
782/782 [==============================] - 30s 38ms/step - loss: 0.1848 - accuracy: 0.9388 - val_loss: 1.4365 - val_accuracy: 0.5883

Trying one more time with early stopping.