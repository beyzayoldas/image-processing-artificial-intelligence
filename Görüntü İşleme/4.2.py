{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4c047c7e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "bd476d05",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dir = 'C:\\\\Users\\\\user\\\\source\\\\proje_odev\\\\train'\n",
    "validation_dir = 'C:\\\\Users\\\\user\\\\source\\\\proje_odev\\\\validation'\n",
    "train_happy_dir = os.path.join(train_dir, 'happy')\n",
    "train_sad_dir = os.path.join(train_dir, 'sad')\n",
    "validation_happy_dir = os.path.join(validation_dir, 'happy')\n",
    "validation_sad_dir = os.path.join(validation_dir, 'sad')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3c340bb4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 6273 images belonging to 2 classes.\n",
      "Found 2964 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "train_datagen = ImageDataGenerator(rescale=1./255)\n",
    "validation_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    train_dir,\n",
    "    target_size=(48, 48),\n",
    "    batch_size=32,\n",
    "    class_mode='binary'\n",
    ")\n",
    "\n",
    "validation_generator = validation_datagen.flow_from_directory(\n",
    "    validation_dir,\n",
    "    target_size=(48, 48),\n",
    "    batch_size=32,\n",
    "    class_mode='binary'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d949a889",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential([\n",
    "    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(64, (3, 3), activation='relu'),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(128, (3, 3), activation='relu'),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(128, (3, 3), activation='relu'),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Flatten(),\n",
    "    Dropout(0.4), \n",
    "    Dense(512, activation='relu'),\n",
    "    Dense(1, activation='sigmoid')\n",
    "])\n",
    "\n",
    "model.compile(loss='binary_crossentropy',\n",
    "              optimizer='adam',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "512b7b5d",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "100/100 [==============================] - 10s 99ms/step - loss: 0.5363 - acc: 0.7859 - val_loss: 0.9702 - val_acc: 0.3937\n",
      "Epoch 2/50\n",
      "100/100 [==============================] - 10s 95ms/step - loss: 0.5110 - acc: 0.7912 - val_loss: 1.0463 - val_acc: 0.3937\n",
      "Epoch 3/50\n",
      "100/100 [==============================] - 10s 96ms/step - loss: 0.4999 - acc: 0.7909 - val_loss: 0.9408 - val_acc: 0.3937\n",
      "Epoch 4/50\n",
      "100/100 [==============================] - 9s 92ms/step - loss: 0.5045 - acc: 0.7853 - val_loss: 1.0078 - val_acc: 0.3937\n",
      "Epoch 5/50\n",
      "100/100 [==============================] - 9s 92ms/step - loss: 0.4925 - acc: 0.7922 - val_loss: 1.0156 - val_acc: 0.3937\n",
      "Epoch 6/50\n",
      "100/100 [==============================] - 8s 81ms/step - loss: 0.4931 - acc: 0.7881 - val_loss: 0.9995 - val_acc: 0.3937\n",
      "Epoch 7/50\n",
      "100/100 [==============================] - 9s 89ms/step - loss: 0.4803 - acc: 0.7928 - val_loss: 1.0675 - val_acc: 0.3937\n",
      "Epoch 8/50\n",
      "100/100 [==============================] - 10s 96ms/step - loss: 0.4907 - acc: 0.7828 - val_loss: 1.1172 - val_acc: 0.3937\n",
      "Epoch 9/50\n",
      "100/100 [==============================] - 9s 89ms/step - loss: 0.4752 - acc: 0.7823 - val_loss: 1.0138 - val_acc: 0.4156\n",
      "Epoch 10/50\n",
      "100/100 [==============================] - 9s 93ms/step - loss: 0.4867 - acc: 0.7803 - val_loss: 0.9456 - val_acc: 0.3950\n",
      "Epoch 11/50\n",
      "100/100 [==============================] - 9s 88ms/step - loss: 0.4580 - acc: 0.7941 - val_loss: 0.8632 - val_acc: 0.4437\n",
      "Epoch 12/50\n",
      "100/100 [==============================] - 10s 99ms/step - loss: 0.4484 - acc: 0.8047 - val_loss: 1.1573 - val_acc: 0.3975\n",
      "Epoch 13/50\n",
      "100/100 [==============================] - 10s 99ms/step - loss: 0.4472 - acc: 0.7984 - val_loss: 0.7712 - val_acc: 0.6006\n",
      "Epoch 14/50\n",
      "100/100 [==============================] - 15s 148ms/step - loss: 0.3987 - acc: 0.8331 - val_loss: 0.8103 - val_acc: 0.5681\n",
      "Epoch 15/50\n",
      "100/100 [==============================] - 11s 105ms/step - loss: 0.3953 - acc: 0.8331 - val_loss: 0.8755 - val_acc: 0.5706\n",
      "Epoch 16/50\n",
      "100/100 [==============================] - 11s 112ms/step - loss: 0.3664 - acc: 0.8444 - val_loss: 0.7484 - val_acc: 0.6494\n",
      "Epoch 17/50\n",
      "100/100 [==============================] - 16s 157ms/step - loss: 0.3465 - acc: 0.8556 - val_loss: 0.6328 - val_acc: 0.6831\n",
      "Epoch 18/50\n",
      "100/100 [==============================] - 10s 97ms/step - loss: 0.3476 - acc: 0.8537 - val_loss: 0.5912 - val_acc: 0.7388\n",
      "Epoch 19/50\n",
      "100/100 [==============================] - 11s 113ms/step - loss: 0.3557 - acc: 0.8395 - val_loss: 0.7311 - val_acc: 0.6194\n",
      "Epoch 20/50\n",
      "100/100 [==============================] - 9s 95ms/step - loss: 0.2892 - acc: 0.8800 - val_loss: 0.8534 - val_acc: 0.6775\n",
      "Epoch 21/50\n",
      "100/100 [==============================] - 8s 83ms/step - loss: 0.2827 - acc: 0.8859 - val_loss: 0.7288 - val_acc: 0.6975\n",
      "Epoch 22/50\n",
      "100/100 [==============================] - 8s 83ms/step - loss: 0.2708 - acc: 0.8912 - val_loss: 0.6464 - val_acc: 0.7450\n",
      "Epoch 23/50\n",
      "100/100 [==============================] - 11s 108ms/step - loss: 0.2315 - acc: 0.9081 - val_loss: 0.6150 - val_acc: 0.7500\n",
      "Epoch 24/50\n",
      "100/100 [==============================] - 9s 90ms/step - loss: 0.2491 - acc: 0.8978 - val_loss: 0.6003 - val_acc: 0.7700\n",
      "Epoch 25/50\n",
      "100/100 [==============================] - 11s 111ms/step - loss: 0.2107 - acc: 0.9162 - val_loss: 0.7506 - val_acc: 0.7031\n",
      "Epoch 26/50\n",
      "100/100 [==============================] - 9s 93ms/step - loss: 0.1929 - acc: 0.9319 - val_loss: 0.7273 - val_acc: 0.7462\n",
      "Epoch 27/50\n",
      "100/100 [==============================] - 10s 101ms/step - loss: 0.1954 - acc: 0.9259 - val_loss: 0.6227 - val_acc: 0.7381\n",
      "Epoch 28/50\n",
      "100/100 [==============================] - 9s 94ms/step - loss: 0.1852 - acc: 0.9253 - val_loss: 0.6169 - val_acc: 0.8013\n",
      "Epoch 29/50\n",
      "100/100 [==============================] - 8s 82ms/step - loss: 0.1554 - acc: 0.9431 - val_loss: 0.8903 - val_acc: 0.7212\n",
      "Epoch 30/50\n",
      "100/100 [==============================] - 8s 83ms/step - loss: 0.1629 - acc: 0.9384 - val_loss: 1.0567 - val_acc: 0.7137\n",
      "Epoch 31/50\n",
      "100/100 [==============================] - 8s 79ms/step - loss: 0.1371 - acc: 0.9469 - val_loss: 0.8569 - val_acc: 0.7369\n",
      "Epoch 32/50\n",
      "100/100 [==============================] - 8s 78ms/step - loss: 0.1342 - acc: 0.9550 - val_loss: 0.9019 - val_acc: 0.7412\n",
      "Epoch 33/50\n",
      "100/100 [==============================] - 9s 88ms/step - loss: 0.1292 - acc: 0.9503 - val_loss: 0.7384 - val_acc: 0.7738\n",
      "Epoch 34/50\n",
      "100/100 [==============================] - 11s 106ms/step - loss: 0.1124 - acc: 0.9569 - val_loss: 0.9675 - val_acc: 0.7431\n",
      "Epoch 35/50\n",
      "100/100 [==============================] - 10s 100ms/step - loss: 0.1222 - acc: 0.9544 - val_loss: 0.7315 - val_acc: 0.7744\n",
      "Epoch 36/50\n",
      "100/100 [==============================] - 10s 97ms/step - loss: 0.0875 - acc: 0.9681 - val_loss: 1.2360 - val_acc: 0.7219\n",
      "Epoch 37/50\n",
      "100/100 [==============================] - 10s 103ms/step - loss: 0.0983 - acc: 0.9634 - val_loss: 1.0258 - val_acc: 0.7550\n",
      "Epoch 38/50\n",
      "100/100 [==============================] - 10s 100ms/step - loss: 0.0830 - acc: 0.9712 - val_loss: 1.1255 - val_acc: 0.7462\n",
      "Epoch 39/50\n",
      "100/100 [==============================] - 9s 90ms/step - loss: 0.0603 - acc: 0.9788 - val_loss: 1.0598 - val_acc: 0.7619\n",
      "Epoch 40/50\n",
      "100/100 [==============================] - 10s 103ms/step - loss: 0.0828 - acc: 0.9694 - val_loss: 0.9718 - val_acc: 0.7869\n",
      "Epoch 41/50\n",
      "100/100 [==============================] - 9s 95ms/step - loss: 0.0997 - acc: 0.9631 - val_loss: 1.1948 - val_acc: 0.7388\n",
      "Epoch 42/50\n",
      "100/100 [==============================] - 9s 86ms/step - loss: 0.0530 - acc: 0.9788 - val_loss: 1.0333 - val_acc: 0.7769\n",
      "Epoch 43/50\n",
      "100/100 [==============================] - 9s 92ms/step - loss: 0.0713 - acc: 0.9741 - val_loss: 0.8445 - val_acc: 0.7950\n",
      "Epoch 44/50\n",
      "100/100 [==============================] - 10s 97ms/step - loss: 0.0814 - acc: 0.9706 - val_loss: 0.9975 - val_acc: 0.7738\n",
      "Epoch 45/50\n",
      "100/100 [==============================] - 10s 96ms/step - loss: 0.0555 - acc: 0.9825 - val_loss: 1.0597 - val_acc: 0.7869\n",
      "Epoch 46/50\n",
      "100/100 [==============================] - 9s 87ms/step - loss: 0.0365 - acc: 0.9916 - val_loss: 1.1313 - val_acc: 0.7769\n",
      "Epoch 47/50\n",
      "100/100 [==============================] - 10s 99ms/step - loss: 0.0441 - acc: 0.9831 - val_loss: 1.0766 - val_acc: 0.7825\n",
      "Epoch 48/50\n",
      "100/100 [==============================] - 10s 103ms/step - loss: 0.0481 - acc: 0.9841 - val_loss: 1.2960 - val_acc: 0.7606\n",
      "Epoch 49/50\n",
      "100/100 [==============================] - 10s 96ms/step - loss: 0.0393 - acc: 0.9869 - val_loss: 1.2006 - val_acc: 0.7700\n",
      "Epoch 50/50\n",
      "100/100 [==============================] - 9s 93ms/step - loss: 0.0654 - acc: 0.9766 - val_loss: 1.2475 - val_acc: 0.7325\n"
     ]
    }
   ],
   "source": [
    "history_dropout = model.fit_generator(\n",
    "    train_generator,\n",
    "    steps_per_epoch=100,\n",
    "    epochs=50,\n",
    "    validation_data=validation_generator,\n",
    "    validation_steps=50\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5a946a94",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEWCAYAAACJ0YulAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJztnWeYFMXWgN/DkgUlKUoGRTIs8YKSREVQFFEUEBOKCKJivBfTBbNe9TOHiwoGUMQIehVFRFlMiIJECQLqSlpyDgvn+1E9s7OzM7MzuzMbz/s880x3dXX1qZ6eOl3nVJ0SVcUwDMMwAErktwCGYRhGwcGUgmEYhuHHlIJhGIbhx5SCYRiG4ceUgmEYhuHHlIJhGIbhx5RCHiMiSSKyW0TqxDNvfiIiJ4lI3Mc2i8gZIrI2YH+5iHSJJm8OrvWKiNyZ0/ON6BCRe0TkpQjHh4rI13kokhFEyfwWoKAjIrsDdssDB4DD3v61qjoplvJU9TBQId55iwOq2ige5YjIUOBSVe0eUPbQeJRtREZV7/dti8hJwEpVlZyWJyKpQFXcfzIdWAK8DrysBXASlifvpar6dX7LEg5TCtmgqv5G2XsTHaqqX4bLLyIlVTU9L2QzjOwoJs9jb1X9WkQqAd2Bp4D2wDWhMotIkvfCZYTAzEe5REQeEJF3RORtEdkFXCoinUTkBxHZLiLrReQZESnl5S8pIioi9bz9id7xz0Rkl4h8LyL1Y83rHe8tIitEZIeIPCsi34rIlWHkjkbGa0VklYhsE5FnAs5NEpEnRWSLiPwO9Ipwf+4WkclBac+LyP9520NFZJlXn9+9t/hwZaWKSHdvu7yIvOnJtgRoG+K6q71yl4jIeV56C+A5oItnmtsccG/HBpw/3Kv7FhH5SEROiObexHKfffKIyJcislVENojIPwOuc493T3aKyDwRqRHKVCcic3y/s3c/Z3vX2QrcLSINRWSWV5fN3n07JuD8ul4d07zjT4tIWU/mJgH5ThCRvSJSNcxv08rbvtK7RycH3Mv3vO0HROQ177TZXtpu79M+ozh50rv+ahHpGe7+BqKq21X1I2AQcLWINPYKm+g9c9NFZA/ut6/kpaeJyFoRuUNEJOgeviDuv7RMRE4LqGstEfnE+81WishVAceCnyO/WVNE3gZqAJ959b0lmnrlOapqnyg/wFrgjKC0B4CDwLk4JVsO95byD1xPrAGwArjey18SUKCetz8R2Ay0A0oB7wATc5D3OGAX0Nc7dgtwCLgyTF2ikXEqcAxQD9jqqztwPa6bXgvXdZ/tHqWQ12kA7AaOCih7E9DO2z/XyyNAD2Af0NI7dgawNqCsVKC7t/048DVQGagLLA3KezFwgvebXOLJUN07NhT4OkjOicBYb7unJ2MyUBZ4AfgqmnsT430+BtgIjALKAEcDHbxjdwC/Ag29OiQDVYCTgu81MMf3O3t1SwdGAEm45/Fk4HSgtPecfAs8HlCfxd79PMrLf6p3bBzwYMB1bgU+DFPPt4BR3vZ44HfgmoBjNwT8X17ztkPVZSjuub3Kk/8G4K8I/0n/MxGUvi7g+hOBbUAn716W8WT6AKjo/S6rgCuC7uGNuP/SJcB2oJJ3/FvgWe/ZaIP7T3YLfo6ye4YL6iffBShMH8Irha+yOe824F1vO1RD/1JA3vOAxTnIexWQEnBMgPWEUQpRytgx4PgHwG3e9mycGc137OzgP3dQ2T8Al3jbvYEVEfJ+Aoz0tiMphT8DfwvgusC8IcpdDJzjbWenFF4HHgo4djTOZl0ru3sT432+DJgXJt/vPnmD0qNRCquzkaE/8JO33QXYACSFyHcqsAYQb38BcEGYMq8FPvC2V3py+F5Y/iZD0UejFH4LuvcKVAtz3XBKYR7wr4DfdnzAsVK4Rv/kgLSRwJcBMvzlq7eX9guuB1Ifp7SOCjj2GPBK8HOU3TNcUD9mPooPfwXuiEhjEfmfZw7YCdwHVItw/oaA7b1Edi6Hy1sjUA51T2BquEKilDGqawF/RJAX3FvZIG/7EsDvnBeRPiLyo9cV3457S490r3ycEEkGz4Txq2eC2A40jrJccPXzl6eqO3FvmjUD8kT1m2Vzn2vj3lBDURunGHJC8PN4vIhMEZG/PRleC5JhrYawsavqt7jGs7OINAfqAP8Lc81vgK4iUtM75z2cmeYk3Bv1ohjkD763EPuAi5q4HpyPwHtyHK4XEvjM/EHm3zfV+w8FHq/hfTar6p4I5xZqTCnEh+BRDv/FvZmepKpHA//GvbknkvW4N1nAGWWJ/KDmRsb1uMbER3ZDZt8BzhCRWjjz1luejOVwjcfDONNOJeCLKOXYEE4GEWkAvIgzoVT1yv0toNzsRqWsw5mkfOVVxJmp/o5CrmAi3ee/gBPDnBfu2B5PpvIBaccH5Qmu36O4UXMtPBmuDJKhrogkhZHjDeBSXK9miqoeCJVJVX/DKYORwDequh3XKPt6sKHueUJGB4lIR6A6rgcV6lqbcD2/ugFpdcj8+9YiM3Vwz8U6oJqIHBXm3D24UYo+svttChymFBJDRWAHsMdz1F2bB9f8BGgjIueKSEmcnfrYBMk4BbhJRGp6Tsd/Rcqsqhtxf9AJwHJVXekdKoOzc6cBh0WkD872Ha0Md3oOwzo4P4ePCrg/XxpOPw7F9RR8bARqSYDDN4i3cY7KliJSBqe0UlQ1bM8rApHu8zSgjohcLyKlReRoEengHXsFeEBEThRHsohUwSnDDbgBDUkiMozMjVs4GfYAO0SkNs6E5eN7YAvwkDjnfTkROTXg+Js4c9MlOAURidm43+Ebb//roP1gNgHqKfFcIyLHiBtQ8BbORLUsVD5VPYR7GXlIRCqIG6xxM8704+ME73cpKSIDcQp6uqquwZmmHhKRMiKSDAwho/e7ADhHRCqLG5xwY9DlN+J8GAUWUwqJ4VbgCpzj97+4N+WE4jW8A4D/w/3JTwTm494Q4y3ji8BMnEngJ9wfLDvewtlX3wqQeTvuz/gh7q2yP065RcMYXI9lLfAZAQ2Wqi4EngHmenkaAz8GnDsDZ/feKCKBpgrf+dNxZp4PvfPrAIOjlCuYsPdZVXcAZwIX4hrIFUA37/BjwEe4+7wT5/Qt671xXwPciXNwnhRUt1CMATrglNM04P0AGdKBPkATXK/hT9zv4Du+Fvc7H1TV77K5zjc4BTQ7zH4mVHUXTuH+6Jn52mVTfjg+Ezef6E9gNO7eZTfv5DrcAJE1npyvk1npfQc0wz2XY4ELVXWbd2wAbgDABtyzf6eqzvKOvQYsw5mUpgOZRt4BDwH3evW9KaZa5hESuldnFHY8c8A6oL+qpuS3PEbhRUTewDmvx+a3LHmBhJjcWJywyWtFCBHphTMH7McNaUzHvS0bRo7wTDt9gRb5LYuRN5j5qGjRGViNMyv0As4P5xg0jOwQkYdxcyUeUtU/81seI28w85FhGIbhx3oKhmEYhp9C51OoVq2a1qtXL7/FMAzDKFT8/PPPm1U10jB1oBAqhXr16jFv3rz8FsMwDKNQISLZRR4AzHxkGIZhBGBKwTAMw/CTMKUgIuNFZJOILA5zXMTFfF8lIgtFpE2iZDEMwzCiI5E9hdeIsPgKLoRyQ+8zDBc6wTAMw8hHEqYUVHU2mUPXBtMXeEMdPwCVvABShmEYRj6Rnz6FmmSOcZ5KmFDPIjJM3HKE89LS0vJEOMMwjGiYNAnq1YMSJdz3pEnZnVGwyU+lECpmfsjp1ao6TlXbqWq7Y4/NdpitYRhGnjBpEgwbBn/8Aarue9iwwq0Y8lMppJJ5kZRauKiehmEYhYK77oK9ezOn7d3r0mOloPQ48lMpTAMu90YhdQR2qOr6fJTHMAwjLKEa7T/DhAn888/wjXyo9ALV40jU4s+41avW4xa5TgWuBoYDw73jAjyPW4d2EdAumnLbtm2rhmEUXSZOVK1bV1XEfU+cmP9lTZyoWr68qmuy3ad8edWqVTOn+T5Vq4bOP2JEbOXUrZvzugcDzNNo2u5oMhWkjykFwyi6hGt8c9KYx7OsunVja/zDNfJJSaHTw31EMuqSW+UWrVKwGc2GYRQY4mmjj1RWrPb7cGairVth3DioWxdE3Pe4cS49FIcPx1aHOnXywbQUjeYoSB/rKRhG0SDU269I5DfmWAhXlu9tPlQPItwbebieQjjzTrj84XoK4XocPnniYVrCzEeGYeQlsZg4YrXR+xrAWK4Rz4Y5VlNUuPzhfAqRFFK8FKUpBcMw8oxYG81YbfTxbJhjsennVCFFyh9rOdZTMKVgGIWOSA1XrGainJhwYmmAw5UT7pMT01U8iZfDPFqlUOjWaG7Xrp3aIjuGUbAoUcI1V6EoXz6zw7d8eShXDrZsyZq3bl1YuzY+1xg3DgYPzprX57iNh0x5xaRJzkH+55/O+fzgg6HrFgkR+VlV22WXz0YfGUYxINbRNrHmr1MndHpSUugRQOAa4kDKl3eNXbyuEW7E0uDBoUcMPf107DLlFYMHO8V05Ij7jlUhxEQ03YmC9DHzkWHERrxs8ZGcobHa7yOZiWKtRzzNPvGcOFfQwHwKhmGoxm6Lz4kTWDW2snI6UzcvrlFUMaVgGMWQWJy6od60czM6J5w88ZpVnJ/XKApEqxTMp2AYBZSc+AFCzXytUiV0/nC2+KSk2OQMN9sXwtvv42kTz4trFCds9JFhFEDCjZCJ1NjVq+cUQTBVq8K+fVnLClYIgcRrxJBRcLDRR4ZRiMlJ3J5Y4/PUrRs6f+DxwjI6x4gfJfNbAMMwshKugfeZhHwKw7cPbshmqJ5CnTqudxGqhxGqN+IbAx+uR5Lb8fJGwcZ6CoYRZ+KxglZOxuQ/+GBsb/I5scXn6Xh5I18wpWAYcSReYY7DNfDhQi//+ac18kZ8MKVgGHEkJ76AUOnhGvhwfgBfz8IaeSO3mE/BMOJIrL6Ab7+F118P7SOI1Q9gGPHAegqGEUdi9QWMGxefuD3WIzDihSkFw8gB4UxBsfoCIvkIwmEmIiORmFIwjBiJ5EyO1RcQbvZwuB6HYSQaUwqGESPZLS4f6k0+XA9i2DCbEGYULEwpGAaxzS0IZ9rJSQygF14wH4FRsLDYR0axJ9Y4Q+FiDFkMIKMgUyBiH4lILxFZLiKrRGR0iON1RWSmiCwUka9FpFYi5TGMUGRnDgom1pnDhlGYSJhSEJEk4HmgN9AUGCQiTYOyPQ68oaotgfuAhxMlj2GEI5I5KJaJZWbyMYoCiZy81gFYpaqrAURkMtAXWBqQpylws7c9C/gogfIYRkjCBZKrUiV88LlIAeMMozCTSPNRTeCvgP1ULy2QX4ELve1+QEURqZpAmYxiTqg3/3DmIIjNrGQYRYFEKgUJkRbs1b4N6CYi84FuwN9AepaCRIaJyDwRmZeWlhZ/SY1iQbj5BRDaHLR1a+hyIo0yMozCTiKVQipQO2C/FrAuMIOqrlPVC1S1NXCXl7YjuCBVHaeq7VS13bHHHptAkY2iQqgeQSSHcqi5BeEmkNnEMqMok0il8BPQUETqi0hpYCAwLTCDiFQTEZ8MdwDjEyiPUUwI1yMI5TeA8G/+NsrIKI4kTCmoajpwPfA5sAyYoqpLROQ+ETnPy9YdWC4iK4DqgP3djFwTrkcQa0gJG2VkFEds8ppRaPGZhIKXhixRwvUQQhFqQXpr6I3iQIGYvGYYiSJSULpwb/7hFqQ3hWAYGVhPwSiURAo18eCDsYWtMIzigPUUjCJNpFnI5gswjJxjy3EahZJws5AD1yo2JWAYsWM9BaNQYsNFDSMxmFIwCiVmIjKMxGBKwchzYlnQJhK2VrFhxB9TCkaeEmkoabyUhWEYOccczUaeEm628ahRsG9f+DDVhmHkDdZTMPKUcENJt2wJH6zOehCGkXeYUjASQriGPNYIo4HB7ILNTYZhxB+b0WzEHZ/fINSMYgh9rFw511sIJikJDh/Oml63rnMuG4YRHdHOaDafghF3Iq1b4GvIgwPZQWhlEVyOD1voxjASg5mPjLgTKQQFhB5KGm7eQd26ocuyhW4MIzGYUjCiIpyPIFR6TlcsC6UsbOayYeQtphSMbAk3t+C660Knn312/Bpym7lsGHmLOZqNbAkXpjqSE/jBB0MvgGMYRv4QraPZlIKRLZFWMguFiDMBGYZRcLD1FIy4Ec4XEOuax4ZhFHxMKRjZEs7ZO2yYOYENo6hhSsHIlnDO3hdeMCewYRQ1zKdgGIZRDDCfgmEYhhEzphQMwzAMPwlVCiLSS0SWi8gqERkd4ngdEZklIvNFZKGInJ1IeYwMLBy1YRihSJhSEJEk4HmgN9AUGCQiTYOy3Q1MUdXWwEDghUTJY2Rgq58ZhhGOREZJ7QCsUtXVACIyGegLLA3Io8DR3vYxwLoEymN42OpnBY8ZM9z3mWfmrxyGkUjzUU3gr4D9VC8tkLHApSKSCnwK3JBAeYod4d76c7L6mZE4VGHoUBg5Mr8lMYzEKgUJkRY8/nUQ8Jqq1gLOBt4UkSwyicgwEZknIvPS0tLiKmQs0T9zUk48z4klfyQTUawzjm3tgsSyeLG7xytXwsaN+S1N8WHRInj33fyWogCiqgn5AJ2AzwP27wDuCMqzBKgdsL8aOC5SuW3bttVYmThRtW5dVRH3PXFiRnr58qqu2XSf8uVVR4wIne47L1T54fLHeu2cXCMUdetmzuv7+GQIVVbVquHPMRLHQw9l3Ov3389vaYoHKSmqFSu6e/7CC/ktTd4AzNNo2u5oMuXkg/NXrAbqA6WBX4FmQXk+A670tpvgfAoSqdxYlUKkxjRcw5mUFLlBDW7kw5VTtWrs1471GuEabJHQ+UUy7kvwNWJVPEZ86NRJtWVL1bJlVW++Ob+lKfp8+aV7rhs1Uj3rLNUSJVQ/+ii/pUo8+a4UnAycDawAfgfu8tLuA87ztpsC33oKYwHQM7syY1UKkRrTcA1npE+oRjPWMrK7dizXEImPEvERrmdjJIZNm9y9HjtWtWtX1fbt81uigs2qVarnn6/6++85O//TT1XLlFFt3lx1wwbV3bvdPS9XTvX77yOfm5Ki2r276uOPq+7bl7Pr5ycFQikk4hOrUoj0xhxrTyHW9EgNebyuHa43EqsJzMgfXn/d/Tbz5qneeaf7nXfvzm+pCib79qm2bu3u1w03xH7+Bx+oliql2qaNalpaRvrGjaoNGqhWq6a6YkXW8w4cUB092v1vjznGXb9WLdVXX1U9dCj89Q4eVJ01S3XRothlTQSmFDxyYlsP16DG2oOIZKMPd+14X8Pe+gs2F12kesIJqocPu7dYUJ05M7+lKphcd527P02aqFaqpLp3b/Tnvv22U7gdO6pu25b1+IoV7r/UoIFTEj4WL1ZNTnbXHTpUdedO9/u0b58hywcfqB454vJv3qz65puqF1+sevTRLk/FiqoLF+au7vHAlIJHdnbySI7gWEwyObHRx+Ma2fkOjILLgQOu4Rg61O1v3+5+t3vvzV+5CiLvvOOe69tucz4BUJ00KbpzJ050foOuXV2jHo7vv3dmpPbtVXftUn3qKWdqOvbYrD6HI0fcoIBGjZwsHTqodu7srgOq1aurXn21k7FGDdXatVXXrct5/eOBKYUA4vXGnBNHbKzXjucoI6Ng42vcpk7NSGvVSvXMM/NWjkWLVL/9Nvr869apXnut6tatiZMpkJUr3dt2x47OJHP4sGr9+qqnnZb9uVu3OsXbtavqnj3Z55861TXsvh74Oec430M4Dh1yZqSTTnI9invuUZ0718no45dfVI86SrVt2/w1DZpSSBB5YZKJ5Ro2Yijx7N+f2aQQL266yb2JBjYUI0eqVqgQ2VYdT3budOarcuVU16yJ7pyLL3bP2ZNPJlQ0VXV+hORk1SpVVP/4IyP9gQecDCtXRj5/7FiX79dfo7/mSy85/8JLL2WYhXLLxx87ZdO3r2p6evh8hw6pLl8en2sGY0qhGGG+g8SxbZtrlGrWjPxnjpUjR1RPPFG1d+/M6W+/rX7Hc15wxx3uemXLqp57bvb5Z8xQ/8CHvPgrjhjhrvfxx5nTU1NdI3vHHeHP3b7d+R769Yv9uvFSBoE884yrS6hhx4cOqb7xhmrDhi7P9Onxv74pBcPIJbt3q55ySkYP7Icf4lf2b7+5Mp9/PnP6X3+59Keeit+1wvH776qlS6teeqkbZgmRx+sfOOBs6A0aZEy4W7YscfJNnuyucfvtoY/36eN6OeF6Vfff787/5ZfEyRgrN97oZHruObd/6JAbgXbSSS69VSv3AvKPf8RfMZlSMIxcsH+/as+e7m30pZcy5hLEi8cec/++tWuzHqtbV7V///hdKxwXXOBMjampzlbfooVziO7aFTr/I484mT/5xPkVSpRQvfvuxMi2eLHzI5xyipMtFB995OSZNi3rsZ07VStXjq73k5ekpzuZSpRQveuuDGWQnKz64YfOF/Hf/7q0Tz+N77VNKRhGDjl0yDWY4JyIqu7NrWPH+F2jWzfXCIdi8GDV449PjAnDx8yZrn4PPJCRNmdO+DfzP/90CqRv34y0M890Dt94y/nZZ845XL26u244Dh509+m887Iee/hhV5e5c+MrWzzYtStjvoVPGQTewwMH3ItBhw7xvbemFAwjBxw+rHrllZrFkTpmjHu727w599fYutXZ5O+8M/TxF19011+1KvfXCsWhQ04h1auXdaz/1VerliyZdcLVRRc5v0OgM9o38S6akUt//KH699+R8xw54sxmJUo4M0qoXlQwo0e7exlY9q5dzlEc7K8pSGzZ4ia2hWv0x42Lf2/BlIJhxMiRIxk232BT0fffu/TJk3N/HZ8z+bvvQh9ftMgdf+213F8rFC+84Mp/772sxzZvdsMxO3fOGFb5xRcu//33Z867c6cbtTRiROTrbdvmGumSJVWvuEJ1yZKseQ4ccPM1wIWxCGfCCmbFCnfOQw9lpPlMc+Hub2EgEb2FuCkF4HqgcjSF5cXHlIIRb/bsUf3884wews03Z/0jpqc7G/WQIbm/3uDBrpEMN5rp8GF3Ld+ktniyZYtr9Lt3D9/YvPKKuw/jxzvfysknO9t3qHg/Awe64aIHDoS/5q23Op/MFVc4JQLO5OPrYaSlOXMauN5T4Bj/aOjWzY3kOnzY/ZbHHZf3cz0Sga+38L//xae8eCqFB4BVwBSgV3ZRTBP9MaVg5Jb0dNUff1R98EE3Aap0afdPKFXK9RTCNZYDBrjRLtm9uR054v7IoSY9HTrkGvzLL49cRp8+qo0bR1efWLjxRmeeWbAgfJ7Dh52Dt2pV518AZ+cPxSefaFhnr6qbR1CqVIYyTUtzprgqVdx5nTu70UxlyuR8KPWbb7qyvvpK9f/+z23PmZOzsgoSvt5C+/bx6S3E1XyEWzDnLGCypyAeAk6M5tx4f0wpGLlh717X2PqGmSYnu9AJ06dnP9t0wgSNaiLUxx+7fKVLu8Y/cM7B7Nnu2JQpkcvwjfTZtCmqakXFkiXO/j58ePZ5f/01IwhjpHH+Bw+6Xs/FF4c+3q+fm80bHOJh927nP6hd2zmLczPcd+9eF6iuXz9XVo8eOS+roPHyy/HrLcTdpwC0Ap4CfgNeBOYD/4n2/Hh9TCkYueGNN9TvRI61wV23zp376KOR851+uouief31bnYyqJ56qlMEt9zibOs7dkQuwzcSKLdx/g8ccA3uE08453KlStHX+847Xf7sHL4jRzondHCdZs3SLCOcgjl0KD5hqEeOzFD0X3+d+/IKCgcOuAEB8egtxNN8dCPwM/A5cBFQyksvAfwezUXi+TGlULS5/HLVZ59NXPldujj7eE7/YK1aRY65s3Ch+1c98ojb377dKaAGDTIardNPz/46+/c7k8qtt8Yu4zffuDHw3bpl2PDByZBdDyWYaOIFffedK3/ChIy09HQ37LJOndiimeaU+fOdDF27Jv5aeU28egvxVAr3AXXDHGsSzUXi+TGlUHT5/Xf3RHbrlpjyly2L7k0/Ev/8p7ORh4u2OXSoa4i3bMmcnp7ugq316xf9n7tzZzc/IhZee039YSjatXPxld59N7EROo8ccQrnjDMy0l591cnx1luJu24wzz4bemRTYefgQddbaNcud72FeCqFjkDFgP2KwD+iKTwRH1MKRRefHb169cSUf+utznQTKepldnz1lWaJbOojLc2ZUa69NuflBzJ6tJM3mrd1VWcmKlPG9WSiHdIZL+65x40w+vtvpzCPP95N9kvkBLzihG9E2Cef5LyMaJVCCbLnRWB3wP4eL80w4so777jvjRth27b4ln3gALz2GvTtC9Wr57ycU0+Fo46C6dOzHhs3DvbvhxtvzHn5gXTuDOnpMHdu9nnXrYN+/aBGDXj3XahQIT4yRMvgwc5I9fbb8OijsGEDPPkkiOStHEWVyy+Hc8+FcuUSf61olIJ4WgYAVT0ClEycSEZxZOVKmD8fevRw+7/9Ft/yP/oItmyBYcNyV07p0nD66fDZZ64R9HHoEDz/PPTsCU2b5u4aPk45xX3PmRM534EDcOGFsHMnTJ0KVavG5/qx0KgRtGsHL74ITzwBl1wCHTvmvRxFlVKlYNq0jP9HIolGKawWkRtFpJT3GQWsTrRgRvFiyhT3fffd7jveSuHll6FePTjjjNyX1asXrF3rFJmP995zb+ujRuW+fB+VK0Pz5pGVgiqMGAE//ACvvw4tWsTv+rFy6aXw++9u++GH808OI3dEoxSGA6cAfwOpwD+AXL5vGUZmpkxxppkuXdzbeDyVwu+/w8yZcPXVUCKaJz4bzjrLfQeakJ56Ck4+2SmMeNK9O8yYAf37wxdfwJEjmY8/9xxMmAD33ON6C/nJwIFQvjyMHg116uSvLEbOydYMpKqbgIF5IItRTPntN1i4EJ5+GkqWhIYNYdmy+JX/yiuQlARDhsSnvAYNnAKYPt35D374wdn9n3suPkonkPvug7JlnT/k/fehfn245hpXl6VL4eab4bzzYOzY+F43J1SvDn/8kT/mKyN+ZPsIi0hZERkpIi+IyHjfJy+EM4oHU6Y4h2T//m6/SZP49RQOHXJv0uecAzVrxqdMcD2Cr792juWnnoJjjoErrohf+T4qV4bHHoPUVOfErVcP7rwTatd2yqDZ2KJ1AAAgAElEQVRRI3jzzfgro5xSrZo5lws70TxKbwLH48JcfAPUAnYlUiijePHOO85sVKOG22/cGFavdg7U3PLJJ2400zXX5L6sQHr1gn37XEP93nswdGhiR/yUKePMM199BcuXux5Cq1bOsXz00Ym7rlH8iEYpnKSq9wB7VPV14BwgKneWiPQSkeUiskpERoc4/qSILPA+K0Rke2ziG4WdJUucGWTAgIy0xo3h8GFYtSr35b/8sushxNvW362ba6hHjXLO3uuvj2/5kTj5ZPjPf+Dbb+Gkk/LuukbxIBqlcMj73i4izYFjgHrZnSQiScDzQG+gKTBIRDIN1lPVm1U1WVWTgWeBD2KQ3SgCvPOOM31ccEFGWpMm7ju3JqQ//nB2/6uvdr6KeFK+vFMMu3bB+ec7s45hFAWiUQrjRKQycDcwDVgKPBrFeR2AVaq6WlUP4iKs9o2QfxDwdhTlGkUEVedP6NYNjj8+I/3kk913bpXCeM/zddVVuSsnHGef7b5vuikx5RtGfhDx/UlESgA7VXUbMBtoEEPZNYG/AvZ9w1lDXacuUB/4KobyjULOwoUZ9vFAKlRwjtTcKIXDh51SOOssqFs3d3KGY/hwSE52/hDDKCpE7Cl4s5dzai0NNQZBQ6SBG/L6nqoeDlmQyDARmSci89LS0nIojlHQmDLFDRUNNB35aNIkd8NS777bjdi59tqcl5EdZcq4Xo5hFCWiMR/NEJHbRKS2iFTxfaI4LxWoHbBfC1gXJu9AIpiOVHWcqrZT1XbHHntsFJc2Cjo+01GPHhDqJ23c2PUUNNxrRAQeecR9hg93sY4Mw4ieaNxvPovsyIA0JXtT0k9AQxGpj5sNPRC4JDiTiDQCKgPfRyGLUUSYP9+NLvrXv0Ifb9wY9uyBv/+GWrWiL/fFF+GOO1zsneeftzHzhhEr0cxorp+TglU1XUSuxy3OkwSMV9UlInIfLoTrNC/rIGByYNA9o+gzZYobEdSvX+jjjRu7799+i14pTJwII0e6aJKvvVZwJnQZRmEiW6UgIpeHSlfVN7I7V1U/BT4NSvt30P7Y7MoxihY+09EZZ4QPieAblrpsWXRB7KZNgyuvdLGCpkxxUSUNw4idaMxH7QO2ywKnA78A2SoFwwjFu+/CmjVw//3h81Sv7kJHRDMC6auv4OKLoW1bN8O3bNn4yWoYxY1ozEc3BO6LyDG40BeGETMHDrgomi1auLAN4RDJcDZHYuFCFwOoYUO3xkHFivGV1zCKGzmZ57kXaBhvQYziwXPPuV7C55+74aiRaNLE5YvE008738EXX0CVaMbEGYYRkWh8Ch+TMb+gBC5kxZRECmUUTbZsgQcecHGIevbMPn/jxs5hvGOHMyUFk57uzEXnngsnnBB3cQ2jWBJNT+HxgO104A9VTU2QPEYR5v773ZKRjz0WXX7fCKTly6FDh6zHZ892iia/F5cxjKJENErhT2C9qu4HEJFyIlJPVdcmVDKjSLFypZs3cPXVbonJaAgclhpKKbz/vgtMF+8IqIZRnIlmJPe7QOAigIe9NMOImtGjXViI++6L/pwGDdzQ0lDhLo4cgQ8/hN69nWIwDCM+RKMUSnpRTgHwtksnTiSjqDFnDnzwgZu9HBgNNTtKlXLrBYQagfT997B+fei4SYZh5JxolEKaiJzn2xGRvsDmxIlkFCWOHIFbb3Wrqt1yS+znhxuW+sEHULo09OmTexkNw8ggGp/CcGCSiDzn7acCIWc5G0YwU6a4Re0nTICjjor9/CZN4OOP3VrLvlnKqs6fcOaZthSlYcSbbHsKqvq7qnbEDUVtpqqnqGocFko0ijr79ztfQqtWcNllOSujcWM39PT33zPSfvnFrapmo44MI/5kqxRE5CERqaSqu1V1l4hUFpEH8kI4o3Dz8ceu8X7kkewnqoUjcASSj/ffd+Wdd17ocwzDyDnR+BR6q+p23463CtvZiRPJKCrMnu1GBp1+es7LCFYKPtNR9+7hg+kZhpFzolEKSSJSxrcjIuWAMhHyGwYAKSnQqVPuIpZWrAg1a2YMS126FFasMNORYSSKaJTCRGCmiFwtIlcDM4DXEyuWUdjZvt0Fq4vH+sWBI5Def98Fyzv//NyXaxhGVqJxNP8HeABognM2TwcStBS6UVT47jtn6omnUvCZjk45xWIdGUaiiHZtqg24Wc0X4tZTyMWS6kZxICXFrazWsWPuy2rSxMVMmjPH9T7MdGQYiSPsPAURORm3rvIgYAvwDiCqeloeyWYUYlJS3KI38QhB4XM2P/ig+7ZZzIaROCL1FH7D9QrOVdXOqvosLu6RYURk3z43YS0epiPIUAqff+4UTV0zXhpGwoikFC7EmY1micjLInI6IHkjllGYmTvXzUCOl1KoUSNjRTUzHRlGYgmrFFT1Q1UdADQGvgZuBqqLyIsiEsUSKUZxJSXFfXfuHJ/yfEtzgikFw0g00Yw+2qOqk1S1D1ALWACMTrhkRqElJcWtmRDP5TG7d3dK5uST41emYRhZiWmNZlXdCvzX+xhGFtLT3XDUnMY6Csd//uOGpBqGkViiHZJqGFHx66+we3f8/AmBiHm0DCPhmFIw4orPn5AIpWAYRuJJqFIQkV4islxEVolISD+EiFwsIktFZImIvJVIeYzEk5IC9epBrVr5LYlhGDkhJp9CLIhIEvA8cCZuYZ6fRGSaqi4NyNMQuAM4VVW3ichxiZLHiJ5165yTuGzZ2M5TdUqhV6/EyGUYRuJJZE+hA7BKVVd76zpPBvoG5bkGeN4Lx42qbkqgPEaUdOoEgwbFft6KFZCWZqYjwyjMJFIp1AT+CthP9dICORk4WUS+FZEfRCTkO6aIDBOReSIyLy0tLUHiGuAmnf35J3z0EcyYEdu55k8wjMJPIpVCqLEiwYMKSwINge64GEuviEilLCepjlPVdqra7thjj427oEYGmwL6ajfd5IaYRktKChx7LDRqFH+5DMPIGxKpFFKB2gH7tYB1IfJMVdVDqroGWI5TEkY+sWGD+x4yxC1o89JL0Z+bkuImmNnQUcMovCRSKfwENBSR+iJSGhdxdVpQno+A0wBEpBrOnLQ6gTIZ2eBTCtdeC2ecAf/+N2zZkv15f/8Na9aY6cgwCjsJUwqqmg5cD3yOW39hiqouEZH7RMS35PrnwBYRWQrMAm5X1SiaICNRbNzovo8/Hp580q1jMGZM9ueZP8EwigYJG5IKoKqfAp8Gpf07YFuBW7yPUQDw9RSqV3chqkeMgBdecD2HFi3Cn5eSAhUqQHJy3shpGEZisBnNRiY2bIBKlTLmKNx7r9u/6abIsYdSUtxQ1pIJfc0wDCPRmFIwMrFxo+sl+KhSBe67D776CqZODX3Otm2weLGZjgyjKGBKwcjEhg3OnxDItddCs2Zw662wf39GuqrLP2mS2+7aNW9lNQwj/lhn38jEhg3QunXmtJIl4emn3Wikfv2gVClYvdp99u1zeSpUgA4d8l5ewzDiiykFIxMbNmQ2H/k4/XQYPNiZkBo0gIYN4ayz3HaDBtCyJZQrl/fyGoYRX0wpGH727XNDUIPNRz4mTsxbeQzDyHvMp2D4CZyjYBhG8cSUguEncI6CYRjFE1MKhh/rKRiGYUrB8OPrKZhSMIziiykFw49PKVh0csMovphSMPxs3AjVqrl5CIZhFE9MKRh+Qs1mNgyjeGFKwfATbuKaYRjFB1MKhp+NG62nYBjFHVMKBpAR3M6UgmEUb0wpGADs3g1795r5yDCKO6YUDMAmrhmG4TClYAA2cc0wDIcpBQOwuEeGYThMKRiAmY8Mw3CYUjAA11NISoKqVfNbEsMw8hNTCgbglMKxxzrFYBhG8cWUggHYxDXDMBwJVQoi0ktElovIKhEZHeL4lSKSJiILvM/QRMpjhMcmrhmGAQlco1lEkoDngTOBVOAnEZmmqkuDsr6jqtcnSg4jOjZsgKZN81sKwzDym0T2FDoAq1R1taoeBCYDfRN4PSOHqJr5yDAMRyKVQk3gr4D9VC8tmAtFZKGIvCcitUMVJCLDRGSeiMxLS0tLhKzFmu3b4eBBUwqGYSRWKUiINA3a/xiop6otgS+B10MVpKrjVLWdqrY71pYFizs2cc0wDB+JVAqpQOCbfy1gXWAGVd2iqge83ZeBtgmUxwiDhbgwDMNHIpXCT0BDEakvIqWBgcC0wAwickLA7nnAsgTKY4TBZjMbhuEjYaOPVDVdRK4HPgeSgPGqukRE7gPmqeo04EYROQ9IB7YCVyZKHiM8Zj4yDMNHwpQCgKp+CnwalPbvgO07gDsSKYORPRs2QKlSULlyfktiGEZ+YzOaDf9wVAk1NMAwjGJFQnsKRuFgw4bCYTo6dOgQqamp7N+/P79FMYwCS9myZalVqxalSpXK0fmmFAw2bIBatfJbiuxJTU2lYsWK1KtXD7FujWFkQVXZsmULqamp1K9fP0dlmPnIKDSzmffv30/VqlVNIRhGGESEqlWr5qo3bUqhmHP4MGzaVDjMR4ApBMPIhtz+R0wpFHO2bHGKoTD0FAzDSDymFIo5RXni2qRJUK8elCjhvidNyl15W7ZsITk5meTkZI4//nhq1qzp3z948GBUZQwZMoTly5dHzPP8888zKbfCFhM+/PBDHnvsMQA++OADfvvtN/+xzp07s2DBgojnr1q1inLlytG6dWuaNGnCP/7xD958882Eypwd48ePZ4Nv8lA+YI7mYk5Rnbg2aRIMGwZ797r9P/5w+wCDB+eszKpVq/obmbFjx1KhQgVuu+22THlUFVWlRInQ71sTJkzI9jojR47MmYD5SHp6OiVL5n1z0q9fP//2Bx98QIkSJWjcuHFMZTRq1Ij58+cDTkn4yrzssssy5curOo4fP542bdpwfD69qVlPoZhTVOMe3XVXhkLwsXevS483q1atonnz5gwfPpw2bdqwfv16hg0bRrt27WjWrBn33XefP6/v7TU9PZ1KlSoxevRoWrVqRadOndi0aRMAd999N0899ZQ//+jRo+nQoQONGjXiu+++A2DPnj1ceOGFtGrVikGDBtGuXbuQb8Vjxoyhffv2fvlUXUzKFStW0KNHD1q1akWbNm1Yu3YtAA899BAtWrSgVatW3OXdrMA37g0bNnDSSScB8MorrzBw4ED69OlD79692blzJz169KBNmza0bNmSTz75xC/HhAkTaNmyJa1atWLIkCFs376dBg0akJ6eDsD27dupX78+hw8f9p+Tnp5OgwYNANi8eTMlSpTw179Tp06sXbuWV155hZtuuomUlBQ+/fRTbr75ZpKTk/31mTx5cpZ7F4mTTjqJJ554gmeeecb/W1x77bWceeaZDBkyhH379nHFFVfQokUL2rRpw+zZs/33ol+/fpx11lk0atSIBx54wF/mf/7zH5o3b07z5s159tln/c9McnKyP88jjzzCAw88wDvvvMOCBQsYMGBATD3QeGI9hWJOUTUf/flnbOm5ZenSpUyYMIGXXnoJcH/yKlWqkJ6ezmmnnUb//v1pGrSK0Y4dO+jWrRuPPPIIt9xyC+PHj2f06CwLFKKqzJ07l2nTpnHfffcxffp0nn32WY4//njef/99fv31V9q0aRNSrlGjRnHvvfeiqlxyySVMnz6d3r17M2jQIMaOHcu5557L/v37OXLkCB9//DGfffYZc+fOpVy5cmzdujXben//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qqR60UWJkzve5LdSUFU9++yz/Q3+ww8/rLd5N/vQoUO6Y8cOVVVNS0vTE088UY94P0wopfDEE0/okCFDVFX1119/1aSkJL9S2LJli6qqpqena7du3fwNXnDD5NufN2+eNm/eXHfv3q27du3Spk2b6i+//KJr1qzRpKQkf6N60UUX6ZtvvpmlTlu3bvXL+vLLL+stt9yiqqr//Oc/dVTATdm6datu2rRJa9WqpatXr84kaySlICL6/fff+4+Fqt+BAwe0fv36OnfuXFV1je6hQ4f0tdde88uwfPlyDfX/f/jhh/W5557THTt2aLt27bRnz56qqtq9e3f97bffMt33CRMm6MiRI/3njhkzRjt16qT79+/XtLQ0rVKlih4M8UYVfO9VVVu1aqU//PCDzpo1S8uXL++/J5F+D0DnzJmjqqpDhgzRxx57TPft26e1atXS5cuXq6rqZZdd5lckoZRCqPsdTG6UQsImr4lIEvA8cCaQCvwkItNUdWlQvorAjcCPiZKlMPPJJ3DTTdC3L3irDgLwr385889TTzkT0B13wJtvOp/CVVdFLvOqq2DsWNi+vfCaj/IrmoDPhNS3b18mT57M+PHjAfdydeeddzJ79mxKlCjB33//zcaNG8OunjV79mxuvPFGwMUGatmypf/YlClTGDduHOnp6axfv56lS5dmOh7MnDlz6Nevnz+a5wUXXEBKSgrnnXce9evX98flDxc+OjU1lQEDBrB+/XoOHjxI/fr1Afjyyy8zmcsqV67Mxx9/TNeuXf15ogmvXbduXTp27BixfiLCCSec4A8/fvTRRwNw0UUXcf/99/PYY48xfvx4rrzyyizld+nShWeeeYb69etzzjnnMGPGDPbu3cvatWtp1KhRtnGQzjnnHMqUKUOZMmU47rjj2LhxI7Vq1cq2Xhow4qNDhw7+exLp96hdu7Y/Ftall17KM888w5lnnkn9+vU5+eSTAbjiiit4/vnnuemmm7KVIREk0qfQAVilqqtV9SAwGegbIt/9wH+AghM7toAwfz4MHAitW7vlJQPWH0EEnnjCBXm7807nPB4/Hjp1guzCA51wAvjWIbE5CrFx/vnnM3PmTH755Rf27dvnX9xm0qRJpKWl8fPPP7NgwQKqV6+ebTjkUOGk16xZw+OPP87MmTNZuHAh55xzTrblBDZOwfhCR0P48NE33HAD119/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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "train_acc_dropout = history_dropout.history['acc']\n",
    "val_acc_dropout = history_dropout.history['val_acc']\n",
    "epochs_dropout = range(1, len(train_acc_dropout) + 1)\n",
    "\n",
    "plt.plot(epochs_dropout, train_acc_dropout, 'bo', label='Training accuracy with Dropout')\n",
    "plt.plot(epochs_dropout, val_acc_dropout, 'b', label='Validation accuracy with Dropout')\n",
    "plt.title('Training and validation accuracy with Dropout')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "422bed49",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('model_dropout.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6263919e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
