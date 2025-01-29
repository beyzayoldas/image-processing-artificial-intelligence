{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0a123d8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b2bd85ec",
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
   "execution_count": 13,
   "id": "7492c6bf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 12102 images belonging to 2 classes.\n",
      "Found 2964 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Eğitim verilerini yüklemek için veri akışı oluşturma\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    train_dir,\n",
    "    target_size=(48, 48),\n",
    "    batch_size=16,\n",
    "    class_mode='binary',  # İkili sınıflandırma\n",
    "\n",
    ")\n",
    "\n",
    "# Doğrulama verilerini yüklemek için veri akışı oluşturma\n",
    "validation_generator = validation_datagen.flow_from_directory(\n",
    "    validation_dir,\n",
    "    target_size=(48, 48),\n",
    "    batch_size=16,\n",
    "    class_mode='binary',  # İkili sınıflandırma\n",
    "\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "8bb26a8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "#model oluşturuluyor\n",
    "model = Sequential([\n",
    "    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),  #Bu, 32 adet 3x3 filtre ile evrişim yapacak bir evrişim katmanıdır.\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(64, (3, 3), activation='relu'),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(128, (3, 3), activation='relu'),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Conv2D(128, (3, 3), activation='relu'),\n",
    "    MaxPooling2D(2, 2),\n",
    "    Flatten(), #evrişim ve havuzlama katmanlarının çıktılarını düzleştirerek (vektörleştirerek) birbirine bağlar. \n",
    "    #Böylece tam bağlantılı (fully connected) katmanlara giriş olarak kullanılabilir hale gelir.\n",
    "\n",
    "\n",
    "    Dense(512, activation='relu'),\n",
    "    Dense(1, activation='sigmoid')\n",
    "])\n",
    "\n",
    "model.compile(loss='binary_crossentropy',   # İkili sınıflandırma için çapraz entropi kaybı kullanılır.\n",
    "              optimizer='adam',     #genellikle evrişimli sinir ağlarında iyi sonuçlar verir.\n",
    "              metrics=['accuracy'])    #Modelin doğruluğunu takip etmek için doğruluk metriği belirlenir."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "3c2e1311",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "100/100 [==============================] - 6s 59ms/step - loss: 0.6791 - acc: 0.5906 - val_loss: 0.6546 - val_acc: 0.6338\n",
      "Epoch 2/50\n",
      "100/100 [==============================] - 5s 45ms/step - loss: 0.6672 - acc: 0.6119 - val_loss: 0.6510 - val_acc: 0.6350\n",
      "Epoch 3/50\n",
      "100/100 [==============================] - 4s 44ms/step - loss: 0.6716 - acc: 0.5988 - val_loss: 0.6773 - val_acc: 0.6150\n",
      "Epoch 4/50\n",
      "100/100 [==============================] - 4s 42ms/step - loss: 0.6770 - acc: 0.5969 - val_loss: 0.6609 - val_acc: 0.6412\n",
      "Epoch 5/50\n",
      "100/100 [==============================] - 5s 45ms/step - loss: 0.6720 - acc: 0.5956 - val_loss: 0.6428 - val_acc: 0.6412\n",
      "Epoch 6/50\n",
      "100/100 [==============================] - 4s 44ms/step - loss: 0.6676 - acc: 0.6006 - val_loss: 0.6283 - val_acc: 0.6687\n",
      "Epoch 7/50\n",
      "100/100 [==============================] - 4s 42ms/step - loss: 0.6630 - acc: 0.6006 - val_loss: 0.6585 - val_acc: 0.6338\n",
      "Epoch 8/50\n",
      "100/100 [==============================] - 5s 48ms/step - loss: 0.6516 - acc: 0.6231 - val_loss: 0.6347 - val_acc: 0.6488\n",
      "Epoch 9/50\n",
      "100/100 [==============================] - 5s 54ms/step - loss: 0.6431 - acc: 0.6231 - val_loss: 0.6434 - val_acc: 0.6412\n",
      "Epoch 10/50\n",
      "100/100 [==============================] - 5s 55ms/step - loss: 0.6449 - acc: 0.6319 - val_loss: 0.6039 - val_acc: 0.6737\n",
      "Epoch 11/50\n",
      "100/100 [==============================] - 5s 48ms/step - loss: 0.6242 - acc: 0.6444 - val_loss: 0.5735 - val_acc: 0.7087\n",
      "Epoch 12/50\n",
      "100/100 [==============================] - 5s 49ms/step - loss: 0.6083 - acc: 0.6819 - val_loss: 0.5709 - val_acc: 0.7163\n",
      "Epoch 13/50\n",
      "100/100 [==============================] - 4s 44ms/step - loss: 0.5862 - acc: 0.7017 - val_loss: 0.5414 - val_acc: 0.7462\n",
      "Epoch 14/50\n",
      "100/100 [==============================] - 4s 43ms/step - loss: 0.5719 - acc: 0.7013 - val_loss: 0.5128 - val_acc: 0.7625\n",
      "Epoch 15/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.5690 - acc: 0.6994 - val_loss: 0.5178 - val_acc: 0.7488\n",
      "Epoch 16/50\n",
      "100/100 [==============================] - 4s 41ms/step - loss: 0.5500 - acc: 0.7231 - val_loss: 0.5147 - val_acc: 0.7650\n",
      "Epoch 17/50\n",
      "100/100 [==============================] - 5s 45ms/step - loss: 0.5312 - acc: 0.7238 - val_loss: 0.4842 - val_acc: 0.7588\n",
      "Epoch 18/50\n",
      "100/100 [==============================] - 4s 42ms/step - loss: 0.5207 - acc: 0.7469 - val_loss: 0.4937 - val_acc: 0.7675\n",
      "Epoch 19/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.5421 - acc: 0.7238 - val_loss: 0.5301 - val_acc: 0.7400\n",
      "Epoch 20/50\n",
      "100/100 [==============================] - 4s 42ms/step - loss: 0.5350 - acc: 0.7175 - val_loss: 0.4625 - val_acc: 0.7800\n",
      "Epoch 21/50\n",
      "100/100 [==============================] - 4s 44ms/step - loss: 0.5201 - acc: 0.7419 - val_loss: 0.4564 - val_acc: 0.7887\n",
      "Epoch 22/50\n",
      "100/100 [==============================] - 4s 42ms/step - loss: 0.5128 - acc: 0.7319 - val_loss: 0.4418 - val_acc: 0.8025\n",
      "Epoch 23/50\n",
      "100/100 [==============================] - 5s 47ms/step - loss: 0.4783 - acc: 0.7606 - val_loss: 0.4292 - val_acc: 0.8113\n",
      "Epoch 24/50\n",
      "100/100 [==============================] - 5s 50ms/step - loss: 0.4853 - acc: 0.7650 - val_loss: 0.4469 - val_acc: 0.7812\n",
      "Epoch 25/50\n",
      "100/100 [==============================] - 5s 45ms/step - loss: 0.4742 - acc: 0.7606 - val_loss: 0.4286 - val_acc: 0.8125\n",
      "Epoch 26/50\n",
      "100/100 [==============================] - 4s 45ms/step - loss: 0.4681 - acc: 0.7762 - val_loss: 0.4461 - val_acc: 0.7937\n",
      "Epoch 27/50\n",
      "100/100 [==============================] - 4s 43ms/step - loss: 0.4578 - acc: 0.7790 - val_loss: 0.3937 - val_acc: 0.8237\n",
      "Epoch 28/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.4368 - acc: 0.7875 - val_loss: 0.4572 - val_acc: 0.7725\n",
      "Epoch 29/50\n",
      "100/100 [==============================] - 4s 42ms/step - loss: 0.4649 - acc: 0.7719 - val_loss: 0.3954 - val_acc: 0.8137\n",
      "Epoch 30/50\n",
      "100/100 [==============================] - 4s 43ms/step - loss: 0.4505 - acc: 0.7719 - val_loss: 0.4094 - val_acc: 0.8100\n",
      "Epoch 31/50\n",
      "100/100 [==============================] - 4s 44ms/step - loss: 0.4496 - acc: 0.7812 - val_loss: 0.3976 - val_acc: 0.8150\n",
      "Epoch 32/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.4252 - acc: 0.7869 - val_loss: 0.3992 - val_acc: 0.8237\n",
      "Epoch 33/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.4330 - acc: 0.7938 - val_loss: 0.3631 - val_acc: 0.8350\n",
      "Epoch 34/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.4224 - acc: 0.8025 - val_loss: 0.3746 - val_acc: 0.8187\n",
      "Epoch 35/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.4320 - acc: 0.7956 - val_loss: 0.3851 - val_acc: 0.8275\n",
      "Epoch 36/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.4049 - acc: 0.8104 - val_loss: 0.3632 - val_acc: 0.8512\n",
      "Epoch 37/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.4105 - acc: 0.8069 - val_loss: 0.3855 - val_acc: 0.8250\n",
      "Epoch 38/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.4188 - acc: 0.7963 - val_loss: 0.3573 - val_acc: 0.8400\n",
      "Epoch 39/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.3995 - acc: 0.8081 - val_loss: 0.3545 - val_acc: 0.8438\n",
      "Epoch 40/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.4256 - acc: 0.8000 - val_loss: 0.3472 - val_acc: 0.8488\n",
      "Epoch 41/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.4039 - acc: 0.8137 - val_loss: 0.3390 - val_acc: 0.8475\n",
      "Epoch 42/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.3882 - acc: 0.8225 - val_loss: 0.3320 - val_acc: 0.8550\n",
      "Epoch 43/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.4010 - acc: 0.8094 - val_loss: 0.3542 - val_acc: 0.8462\n",
      "Epoch 44/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.3660 - acc: 0.8206 - val_loss: 0.3545 - val_acc: 0.8363\n",
      "Epoch 45/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.3882 - acc: 0.8169 - val_loss: 0.3414 - val_acc: 0.8525\n",
      "Epoch 46/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.3991 - acc: 0.8119 - val_loss: 0.3367 - val_acc: 0.8550\n",
      "Epoch 47/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.3711 - acc: 0.8100 - val_loss: 0.3483 - val_acc: 0.8512\n",
      "Epoch 48/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.3722 - acc: 0.8337 - val_loss: 0.3358 - val_acc: 0.8575\n",
      "Epoch 49/50\n",
      "100/100 [==============================] - 4s 39ms/step - loss: 0.3613 - acc: 0.8300 - val_loss: 0.3432 - val_acc: 0.8538\n",
      "Epoch 50/50\n",
      "100/100 [==============================] - 4s 40ms/step - loss: 0.3792 - acc: 0.8250 - val_loss: 0.3373 - val_acc: 0.8475\n"
     ]
    }
   ],
   "source": [
    "  history = model.fit_generator(   #Bu yöntem, veri artırma (data augmentation) veya \n",
    "    #veri yüklemesiyle birlikte eğitim ve doğrulama verileriyle bir modelin eğitilmesini sağlar.\n",
    "    train_generator,   #Eğitim verilerini sağlayan veri akışı nesnesi.\n",
    "    steps_per_epoch=100,\n",
    "    epochs=50,\n",
    "    validation_data=validation_generator,  #Modelin doğrulama verilerini sağlayan veri akışı nesnesi\n",
    "    validation_steps=50\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7eb0ded4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
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
    "train_acc = history.history['acc']  # 'accuracy' yerine 'acc' kullanıyoruz   -eğitim doğruluğunu temsil eder.\n",
    "val_acc = history.history['val_acc']  # 'val_accuracy' yerine 'val_acc' kullanıyoruz\n",
    "epochs = range(1, len(train_acc) + 1)  #Grafik üzerinde x-eksenini oluşturmak için kullanılacak epoch sayısını belirler.\n",
    "\n",
    "plt.plot(epochs, train_acc, 'bo', label='Training accuracy')\n",
    "plt.plot(epochs, val_acc, 'b', label='Validation accuracy')\n",
    "plt.title('Training and validation accuracy')  #Grafik başlığını belirler, eğitim ve doğrulama doğruluğunu temsil eder.\n",
    "plt.xlabel('Epochs')   #x-ekseni etiketini belirler, epoch sayısını temsil eder.\n",
    "plt.ylabel('Accuracy') #y-ekseni etiketini belirler, doğruluk (accuracy) değerini temsil eder.\n",
    "plt.legend()   #Çizgilerin hangi doğruluk değerini temsil ettiğini gösteren bir açıklama (legend) ekler.\n",
    "plt.show()   #Grafikleri görüntüler."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "63b39c35",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('model_training_validation.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97b8658e",
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
