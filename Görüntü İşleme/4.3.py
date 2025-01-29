{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6a23c5c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "891204b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dir = 'C:\\\\Users\\\\user\\\\source\\\\proje_odev\\\\train'\n",
    "validation_dir = 'C:\\\\Users\\\\user\\\\source\\\\proje_odev\\\\validation'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "28f12fdf",
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
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=40,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    zoom_range=0.2,\n",
    "    horizontal_flip=True\n",
    ")\n",
    "\n",
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
   "id": "d3584a49",
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
   "id": "ac3b638c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "100/100 [==============================] - 8s 77ms/step - loss: 0.6812 - acc: 0.5811 - val_loss: 0.6597 - val_acc: 0.6194\n",
      "Epoch 2/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6699 - acc: 0.5994 - val_loss: 0.6747 - val_acc: 0.6206\n",
      "Epoch 3/50\n",
      "100/100 [==============================] - 6s 65ms/step - loss: 0.6741 - acc: 0.5928 - val_loss: 0.6633 - val_acc: 0.6194\n",
      "Epoch 4/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6656 - acc: 0.6091 - val_loss: 0.6472 - val_acc: 0.6394\n",
      "Epoch 5/50\n",
      "100/100 [==============================] - 6s 65ms/step - loss: 0.6712 - acc: 0.5988 - val_loss: 0.6589 - val_acc: 0.6238\n",
      "Epoch 6/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6679 - acc: 0.5943 - val_loss: 0.6580 - val_acc: 0.6194\n",
      "Epoch 7/50\n",
      "100/100 [==============================] - 6s 65ms/step - loss: 0.6639 - acc: 0.5981 - val_loss: 0.6429 - val_acc: 0.6331\n",
      "Epoch 8/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6549 - acc: 0.6228 - val_loss: 0.6431 - val_acc: 0.6188\n",
      "Epoch 9/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6518 - acc: 0.6294 - val_loss: 0.6715 - val_acc: 0.6256\n",
      "Epoch 10/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6712 - acc: 0.5784 - val_loss: 0.6598 - val_acc: 0.6081\n",
      "Epoch 11/50\n",
      "100/100 [==============================] - 6s 65ms/step - loss: 0.6555 - acc: 0.6094 - val_loss: 0.6356 - val_acc: 0.6200\n",
      "Epoch 12/50\n",
      "100/100 [==============================] - 6s 65ms/step - loss: 0.6534 - acc: 0.6109 - val_loss: 0.6285 - val_acc: 0.6256\n",
      "Epoch 13/50\n",
      "100/100 [==============================] - 6s 65ms/step - loss: 0.6526 - acc: 0.6241 - val_loss: 0.6263 - val_acc: 0.6456\n",
      "Epoch 14/50\n",
      "100/100 [==============================] - 7s 65ms/step - loss: 0.6521 - acc: 0.6088 - val_loss: 0.6302 - val_acc: 0.6412\n",
      "Epoch 15/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.6453 - acc: 0.6316 - val_loss: 0.6209 - val_acc: 0.6450\n",
      "Epoch 16/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6476 - acc: 0.6197 - val_loss: 0.6198 - val_acc: 0.6312\n",
      "Epoch 17/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.6561 - acc: 0.6209 - val_loss: 0.6205 - val_acc: 0.6456\n",
      "Epoch 18/50\n",
      "100/100 [==============================] - 7s 65ms/step - loss: 0.6470 - acc: 0.6172 - val_loss: 0.6273 - val_acc: 0.6475\n",
      "Epoch 19/50\n",
      "100/100 [==============================] - 7s 65ms/step - loss: 0.6368 - acc: 0.6332 - val_loss: 0.6120 - val_acc: 0.6475\n",
      "Epoch 20/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.6428 - acc: 0.6278 - val_loss: 0.6319 - val_acc: 0.6419\n",
      "Epoch 21/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.6521 - acc: 0.6201 - val_loss: 0.6147 - val_acc: 0.6556\n",
      "Epoch 22/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.6511 - acc: 0.6162 - val_loss: 0.6133 - val_acc: 0.6512\n",
      "Epoch 23/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.6274 - acc: 0.6459 - val_loss: 0.6042 - val_acc: 0.6562\n",
      "Epoch 24/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.6361 - acc: 0.6375 - val_loss: 0.6124 - val_acc: 0.6569\n",
      "Epoch 25/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.6342 - acc: 0.6425 - val_loss: 0.5940 - val_acc: 0.6650\n",
      "Epoch 26/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.6371 - acc: 0.6360 - val_loss: 0.5978 - val_acc: 0.6663\n",
      "Epoch 27/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.6364 - acc: 0.6400 - val_loss: 0.5855 - val_acc: 0.6663\n",
      "Epoch 28/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.6224 - acc: 0.6450 - val_loss: 0.5874 - val_acc: 0.6681\n",
      "Epoch 29/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.6305 - acc: 0.6435 - val_loss: 0.5550 - val_acc: 0.7156\n",
      "Epoch 30/50\n",
      "100/100 [==============================] - 7s 71ms/step - loss: 0.6454 - acc: 0.6206 - val_loss: 0.5564 - val_acc: 0.6937\n",
      "Epoch 31/50\n",
      "100/100 [==============================] - 7s 72ms/step - loss: 0.6187 - acc: 0.6571 - val_loss: 0.5885 - val_acc: 0.6906s - loss: 0\n",
      "Epoch 32/50\n",
      "100/100 [==============================] - 7s 71ms/step - loss: 0.6093 - acc: 0.6603 - val_loss: 0.5253 - val_acc: 0.7375\n",
      "Epoch 33/50\n",
      "100/100 [==============================] - 7s 71ms/step - loss: 0.5977 - acc: 0.6663 - val_loss: 0.4989 - val_acc: 0.7488\n",
      "Epoch 34/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5832 - acc: 0.6741 - val_loss: 0.5218 - val_acc: 0.7269\n",
      "Epoch 35/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5679 - acc: 0.6891 - val_loss: 0.4820 - val_acc: 0.7600\n",
      "Epoch 36/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5708 - acc: 0.6850 - val_loss: 0.5207 - val_acc: 0.7269\n",
      "Epoch 37/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5739 - acc: 0.6800 - val_loss: 0.4406 - val_acc: 0.8019\n",
      "Epoch 38/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5478 - acc: 0.7125 - val_loss: 0.4413 - val_acc: 0.7806\n",
      "Epoch 39/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.5535 - acc: 0.7159 - val_loss: 0.4623 - val_acc: 0.7750\n",
      "Epoch 40/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5493 - acc: 0.7094 - val_loss: 0.4428 - val_acc: 0.7788\n",
      "Epoch 41/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5403 - acc: 0.7098 - val_loss: 0.4319 - val_acc: 0.7869\n",
      "Epoch 42/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5314 - acc: 0.7231 - val_loss: 0.4059 - val_acc: 0.8000\n",
      "Epoch 43/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5279 - acc: 0.7259 - val_loss: 0.4177 - val_acc: 0.7963\n",
      "Epoch 44/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.5178 - acc: 0.7412 - val_loss: 0.4124 - val_acc: 0.8050\n",
      "Epoch 45/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5172 - acc: 0.7440 - val_loss: 0.4099 - val_acc: 0.8050\n",
      "Epoch 46/50\n",
      "100/100 [==============================] - 7s 75ms/step - loss: 0.5185 - acc: 0.7387 - val_loss: 0.4189 - val_acc: 0.7944\n",
      "Epoch 47/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.5106 - acc: 0.7391 - val_loss: 0.3881 - val_acc: 0.8181\n",
      "Epoch 48/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.4869 - acc: 0.7542 - val_loss: 0.3990 - val_acc: 0.8113\n",
      "Epoch 49/50\n",
      "100/100 [==============================] - 7s 73ms/step - loss: 0.5036 - acc: 0.7400 - val_loss: 0.3974 - val_acc: 0.8087\n",
      "Epoch 50/50\n",
      "100/100 [==============================] - 7s 71ms/step - loss: 0.4900 - acc: 0.7547 - val_loss: 0.4171 - val_acc: 0.8050\n"
     ]
    }
   ],
   "source": [
    "history_augmentation = model.fit_generator(\n",
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
   "id": "8b638f29",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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W1fT09M1JSVFu3btqrGxsRoREaFffvllxnGVKlXK2L9Zs2aqqpqamqr9+vXTyMhIveWWW7R169a6ePFiVVW96667tEWLFtq0aVN96qmnVFX1lVde0TJlymhERIR27txZVVXr16+vu3fvVlXVF198UZs1a6bNmjXTl156KeN6TZo00eHDh2vTpk316quv1tTU1NNe16xZs7R169YaExOjV155pf7555+qqpqSkqJDhgzRiIgIjYyM1JkzZ6qq6pw5czQ2NlajoqK0a9euqqo6duxYnTBhQsY5mzVrpps2bcqI4e6779aYmBhNTEz0+/pUVX/99Vdt166dRkVFaatWrfTgwYPaoUMHXbZsWcY+7du31xUrVmSKv2fPnhnrYmJi9O9//7uqqo4ZM0bfeeedjPt+7NgxrVu3rtasWVOjo6N1+vTpOnbsWB06dKheccUV2qBBA33llVf8vve+91pVNTk5WatWrap79+4N+N7369dPy5cvr9HR0frggw9m+xnxlZ6erg0aNNANGzbohRdeqEeOHMl4P72fH1XVCRMm6NixYzPuXWRkpLZt21YffPDBjP0mT56sN9xwg/bq1UvDw8P1tdde0xdffFFjYmK0TZs2unfvXlVV3bBhg3bv3l2bN2+uHTp00N9//11VVQcPHqyjRo3Sdu3aaYMGDXTGjBmqqtqmTRutWrWqRkdH67///W/dtGmTdujQQWNjYzU2NlZ/+uknv/vFxcXptddeq6qqe/fu1RtuuEEjIyO1TZs2Ge9hsO9JQf89MMb4d/CgaoUKqvfck//nTk9XjY9XffZZ1RYtVEH13HNVH39cdfv2/L9ebgBLNMgcp8CTrPx65JSs3Xef6hVX5O/jvvuyexuca665JuNL9rnnntMHH3xQVVVPnDihycnJqqq6e/dubdSokaanp6uq/2TtxRdf1KFDh6qq6ooVK7RUqVIZyZr3CzQtLU2vuOKKjC+xrAmDd3nJkiUaERGhhw4d0pSUFG3atKkuXbpUN23apKVKlcpIdm6++WadMmXKaa9p3759GbG+8847ev/996uq6sMPP6z3+dyUffv26a5du7ROnTq6cePGTLFml6yJiP7yyy8Z2/y9vmPHjmmDBg30119/VVWXDJ04cULff//9jBjWrl2r/j4Xzz33nL7++uuanJysLVu21G7duqmqaufOnXXNmjWZ7vvkyZN15MiRGceOHTtW27Vrp0ePHtXdu3frueeeq8ePHz/tGlnvvapqdHS0Llq0KOB7nzW5yu4z4mvBggUZSfCAAQP0s88+U9Xsk7VmzZplJEiPPPJIptfbqFEjPXjwoO7atUurVq2qb731lqqq/u1vf8tI7Lt27arr1q1TVdVFixZply5dVNUlazfddJOePHlSV61apY0aNVJVzZR0qaoePnw4I6lct25dxvuUdT/f5XvvvVfHjRunqqo//PCDRkdH5+o9sWTNmMLpww9dNrJwYeiv9dNPqn37qoqolimjOniwapb/58+a3CRrVg0aYr6dsnmXAAAgAElEQVRVob5VoKrK448/TlRUFFdddRXbtm1j586dAc8zf/58Bg0aBLi5M6OiojK2ffrppzRv3pzY2FhWrVrld5J2XwsXLqRPnz5UqlSJypUr07dvXxYsWABAgwYNMuaZbNGihd85M5OSkujevTuRkZFMmDCBVatWAfD9998zcuTIjP2qV6/OokWL6NSpEw0aNADg3HPPzTY2gPr169O2bdtsX9/atWu58MILadWqFQBVq1aldOnS3HzzzXzzzTecOHGCSZMmMWTIkNPO37FjR+bPn8/ChQu59tprOXToEKmpqSQmJtK4ceMc47v22mspV64cNWvW5Pzzz8/2ffOlnm5Kwb73we43bdo0+vfvD0D//v1zrAo9cOAAKSkpGfOn3pqlkUiXLl2oUqUK5513HtWqVcuYqioyMpLExEQOHTrEzz//zM0330xMTAx33nknO3wahPTu3ZuwsDCaNm0a8N6cOHGCO+64g8jISG6++eYcP7PgPrfeOVi7du3K3r17SU5OBvL+nhhjCt7HH0N4OHj+JIVU+/bw2Wewfj3cdZfr1BAdDVdfDf/9b+HtTVq05hs6Ay+/XDDX7d27N/fffz9Lly7lyJEjNG/eHHATo+/evZvffvuNMmXKEB4eztEcuq6IyGnrNm3axAsvvMDixYupXr06Q4YMyfE8ms2nsVy5chnPS5UqxZEjR07bZ9SoUdx///1cf/31zJ07l3HjxmWcN2uM/tYBlC5dOlN7NN+YK/l0BQr0+gKdt2LFilx99dV89dVXfPrpp347YbRq1YolS5bQsGFDrr76avbs2cM777xDixYtAt4XX1nvUVpaWo7HpKSkkJiYyKWXXhr0ex/MfidPnuSzzz5j1qxZjB8/HlVl7969pKSkBLzH2b3/WV9fWFhYxnJYWBhpaWmkp6dzzjnnsHz58hyPD3Stl156iVq1arFixQrS09MpX758tjEFOpf3M5CX98QYk7309OAnTy9Xzg2pkVu7dsF338HDD7uppc6WRo3g1Vfh7393Q4C89hr07AlNm7qprgYOhCD+LJ01VrIWYpUrV6Zz584MGzYsU8eC5ORkzj//fMqUKUNcXBybN2/O9jydOnVi6tSpAKxcuZL4+HgADh48SKVKlahWrRo7d+5kzpw5GcdUqVKFlJQUv+f68ssvSU1N5fDhw3zxxRd07Ngx6NeUnJxM7dq1Afjggw8y1nfr1o3XX389Y3n//v20a9eOefPmsWnTJgD27dsHuN6SS5cuBWDp0qUZ27MK9PqaNGnC9u3bWbx4MeCSIe8X9PDhw/nrX/9Kq1at/JbklS1blrp16/Lpp5/Stm1bOnbsyAsvvOD3HgS6h7lx6NAh7rnnHnr37k316tUDvvdZrxXMZ+T7778nOjqarVu3kpiYyObNm7nxxhv58ssvqVWrFrt27WLv3r0cO3aMb775BnAlnlWqVGHRokUAp3WCyUnVqlVp0KABM2bMAFwStWLFimyP8ffaLrzwQsLCwpgyZQonPcOQZ3e/fX8H5s6dS82aNalatWquYjfGBGf/fmjbFqpUCe5x0UVuOqjcmjHDzUKQ216g+aV6dTf36KZNrhdpmTIwfDhERRWu2RFKTMlaQRowYAB9+/bN9KU4cOBArrvuOlq2bElMTAxNmjTJ9hx33303Q4cOJSoqipiYGFq3bg1AdHQ0sbGxNGvWjIYNG3L55ZdnHDNixAh69uzJhRdeSFxcXMb65s2bM2TIkIxzDB8+nNjYWL9Vnv6MGzeOm2++mdq1a9O2bduMRGvMmDGMHDmSiIgISpUqxdixY+nbty8TJ06kb9++pKenc/755/Pdd99x44038uGHHxITE0OrVq249NJL/V4r0OsrW7Ysn3zyCaNGjeLIkSNUqFCB77//nsqVK9OiRQuqVq3K0KFDA76Gjh078sMPP1CxYkU6duxIUlKS32StS5cuPP/888TExPDYY48FdX98j1V1PVr79OnDk08+CQR+72vUqMHll19OREQEPXv25JFHHsnxMzJt2jT69OmTad2NN97IW2+9xW233cZTTz1FmzZtaNCgQabj33vvPe644w4qVapE586dqVatWq5e29SpU7n77rt55plnOHHiBP379yc6Ojrg/lFRUZQuXZro6GiGDBnCPffcw4033siMGTPo0qVLRmlq1v1iY2MzzjFu3LiM34GKFStm+kfBGJN/UlJcKdOKFTB2LHhGkwpI1c0m0KcPLF4Muflz8vHHEBkJERFnFvOZKlsWbrsNBg2CuDhITIRSpQo2Jl82kbspdrZv307nzp1Zs2YNYWFWeOzPoUOHMsbze/7559mxYwevvPJKAUcVWvb3wJicpaa6RO2nn1zbrhtuCO64BQuga1e45ho3fEYwf3o3bYKGDeH55+GRR84s7qKo0EzkLiI9RGStiGwQkUf9bK8nInEiskxE4kXkGp9tj3mOWysi3UMZpyk+PvzwQ9q0acP48eMtUcvGf/7zH2JiYoiIiGDBggWMGTOmoEMy5qzbvbugIyhcjh1zpWMLFsCUKcEnagAdO8KLL8KsWfDcc8Ed4+0L5ekfZbIRspI1ESkFrAOuBpKAxcAAVV3ts89EYJmqviUiTYHZqhrueT4NaA1cBHwPXKqqAWuQrWTNGJMd+3tgfC1fDs2bu8btV15Z0NEUvBMn4Oab4auv3EwAw4bl/hyqrirx44/ddE89emS/b0QEnHuuSw5LosJSstYa2KCqG1X1ODAdyJqnK+BtIVwN2O55fgMwXVWPqeomYIPnfMYYY8wZW7nSJQxvvFHQkRS8kyfh9ttdovbaa3lL1MD15pw40bVBu/VWV80ZSEICrF5dcB0LippQJmu1ga0+y0medb7GAYNEJAmYDYzKxbFBKS5t8owxeWd/B0xWSUnu59dfw59/Fmws+WHmTLj8clcqlpsJzNPT3YTo06e7tmP33ntmcVSsCJ9/7hLhvn0DD/3x8cdQurQrzTM5C2Wy5m/ElKx/MQcA76tqHeAaYIqIhAV5LCIyQkSWiMiS3X4aH5QvX569e/faH2pjSjDv2HPBjOVmSo6tW90wDWlpUNQ7Fu/c6RKupUvdsBP168M//uHmyMyOKtx3H0yaBE8+mX+N/Bs1csnYihVu4NmsX8Hp6W579+5uPlCTs1AO3ZEE1PVZrsOpak6vvwA9AFT1FxEpD9QM8lhUdSIwEVybtazb69SpQ1JSEv4SOWNMyVG+fHnq1KlT0GGYQiQpCZo0gXPOgXffPfuDsuan++5zk5MvXw7bt7uG/k89Bc8+C4MHw+jRkHVyFlV47DF4/XU3COzf/56/MfXsCePGuaE/2rQBn8lt+Oknlyw//3z+XrNYC3Zeqtw+cIngRqABUBZYATTLss8cYIjn+WW4hEyAZp79y3mO3wiUyu56/uaANMYYY/yJjVW95ppT81L++GNBR5Q333zj4n/66czrV61SHT5ctVw5t/2661TnznUTm6uq/uMfbv1dd51al99OnlTt1Uu1dOnM837edZdqxYqqKSmhuW5RQS7mBg3pOGueoTheBkoBk1R1vIg87QlwlqfX5ztAZVw158Oq+v88xz4BDAPSgL+p6hy/F/Hw1xvUGGOM8ee881ybqpdfdqPv9+zpquaKkpQUaNYMqlZ1VaBly56+z86d8Oab7rFnD7RoAa1bw1tvuU4FkycHNyZaXh04AK1auZK/336DGjXgwgtdFWhRu9/5LTe9QYv1oLjGGGNMVkePQoUKrl3XmDEwapTrxbh9u0smior77nO9N3/6Cdq1y37fI0fc2Gn//jesXesa9nsb+YdaQoKbuio2Fh54wCXJX38NvXqF/tqFWWEZusMYY4wpdLw9Qet6WkbfcQccPw4ffVRwMeXWokUuUbvnnpwTNXDJ6YgRbriMJUvOXqIGbiiP995zSeXgwS4h7m5D3eeKJWvGGGNKFG+y5u1zEhXlqureeef0nouF0fHjLsGsXdt1IsiNsDBXFXq2EjWv/v1dR4eUFFeqV6bM2b1+UWcTuRtjjClRtnpG8azrM+bAHXe4kqdFi4IrqSpIEya4QX1nzXLt1YqKf/7T3fN+/Qo6kqLHStaMMcaUKFlL1sCV/FSq5ErXCrN161xbu5tvhuuuK+hocqdMGVe6dtFFBR1J0WPJmjHGmBJl61Y3J2XFiqfWVakCAwbAJ5/AwYMFF1t2vLMNVKgAr75a0NGYs8mSNWOMMSXK1q2ZS9W87rjDTY80bdrZjykY770H8+a5atALLijoaMzZZMmaMcaYEiUpKXN7Na9WrVxng8JYFbpjBzz0EHTuDH/5S0FHY842S9aMMcaUKIFK1kTc3Jq//QbLlp39uLLz17+68eEmTiy602KZvLNkzRhjTIlx5Ajs3eu/ZA1g0CAoX75wla7NmgUzZ7r5Pi+5pKCjMQXBkjVjjDElRtYBcbOqXh1uugmmTnXt1wrawYNu4NvISFcNakomS9aMMcaUGP6G7cjqjjtckjRjxtmJKRBVNz3T9u2upM8Gki25LFkzxhhTYvgbEDerjh2hceOCrQpVhQcfhHffdSVqbdoUXCym4FmyZowxpsTwJmu1awfex9vR4Kef3FyaZ1t6Otx7r5t0/a9/heefP/sxmMLFkjVjjDElRlKSm0jcd0Bcf26/3VU7vvvu2YnL6+RJuPNOePNNV6L28svW+9NYsmaMMaYECTRsR1bnnw833AAffgjHjoU+LoC0NBg61CWIY8a4uTQtUTNgyZoxxpgSJNCAuP7ccYcb5uPLL0MbE8CJE27YkClT3Nyf//iHJWrmFEvWjDHGlBhbtwafrF11FYSHh76jwbFjcMstbl7SCRNcqZoxvixZM8YYUyKkpsK+fcFVgwKEhbmpnX74Af74IzQxHT0Kffu60rtXX3U9QI3JypI1Y4wxJUJOA+L6M3SoS9oeeACSk/M3ntRUuO46mDMH3n4bRo3K3/Ob4sOSNWOMMSWCd9iOYEvWwA3x8dxz8M03EBEB336bP7EcOgTXXONK7SZNghEj8ue8pniyZM0YY0yJkJeSNYCHH4ZffoEqVaBHD5dYpaTkPY74eOjWDRYuhI8+giFD8n4uUzJYsmaMMaZECGZA3EBatYKlS13i9t57bq7OH34I/nhV+O9/XZIWHe0Stk8+gVtvzX0spuSxZM0YY0yJkJQENWtChQp5O758eTf22cKFUK6c6y06cqSr0gzk2DFXzRkZCT17wsqVrlp1yxa48ca8xWFKHkvWjDHGlAjBDoibk3btYPlyuP9+eOstiIqCefMy77NnjxsrrX5916O0VCn44ANITIRHH4Vzzz3zOEzJYcmaMcaYEiE3A+LmpEIFePFFl6SFhUHnznDffbBiBdx9N9SrB089Bc2bw3ffueTu9tuhbNn8ub4pWUoXdADGGGPM2bB1K3TokL/n7NjRJWiPPebGSXv1VZeQ3XYbjB4NzZrl7/VMyWTJmjHGmGLv8GHYvz9/qkGzqlTJJWk33wy//QYDBkCtWvl/HVNyWbJmjDGm2MvrsB250bGjexiT36zNmjHGmGIvLwPiGlNYWLJmjDGm2DsbJWvGhIola8YYY4q9MxkQ15iCZsmaMcaYYi8pCc47zw1sa0xRY8maMcaYYm/rVqsCNUWXJWvGGGOKvaQk61xgii5L1owxxhR7VrJmijJL1owxxhRrhw7BgQNWsmaKLkvWjDHGFGs2bIcp6ixZM8YYU6zZgLimqLNkzRhjTLFmJWumqLNkzRhjTLFmA+Kaos6SNWOMMcVaUhKcfz6UK1fQkRiTN5asGWOMKdZs2A5T1FmyZowxpljbutU6F5iizZI1Y4wxhd6SJbBnT96OTUqykjVTtFmyZowxplBLTYWOHeGJJ3J/bEoKJCdbyZop2ixZM8YYU6j9/DMcPQo//JD7Y23YDlMcWLJmjDGmUJs71/38449Tw3AEywbENcWBJWvGGGMKtbg4qFHDPfcmbsGykjVTHFiyZowxptA6dAh+/RX+8heoXt0lbrnhLVm76KL8j80UrKlTITwcwsLcz6lTC+c580Ppgg7AGGOMCeTnnyEtDa68Etaty33J2tatUKuWDYhb3EydCiNGuM4nAJs3u2WAgQMLzznzi5WsGWOMKbTi4qB0aWjfHrp0gU2b3JdosGzYjuLpiSdOJVVeqal56zEcynPmF0vWjDHGFFpxcdC6NVSuDJ07n1oXLBsQt3jasiV36wvqnPnFkjVjjDGFUkqKGwy3Sxe3HBHhOhrkpirUStaKp3r1cre+oM6ZX0KarIlIDxFZKyIbRORRP9tfEpHlnsc6ETngs+2kz7ZZoYzTGGNM4bNwIZw8eSpZCwtzpWtxcaCa8/EHD7qHlawVP+PHQ8WKmddVrOjWF6Zz5peQJWsiUgp4A+gJNAUGiEhT331UdbSqxqhqDPAa8LnP5iPebap6fajiNMYYUzjFxUGZMtCu3al1nTu7aqlNm3I+3obtKL4GDoSJE6F+fRBxPydOPLOOAKE4Z34JZW/Q1sAGVd0IICLTgRuA1QH2HwCMDWE8xhhjipC5c6Ft28ylHd5StrlzoWHD7I+3AXGLt4ED8z+RCsU580Moq0FrA75jTSd51p1GROoDDYAffVaXF5ElIrJIRHqHLkxjjDGFTXIy/PbbqeTMq2lTOO+84DoZeJM1K1kzRV0oS9bEz7pArQz6AzNV9aTPunqqul1EGgI/ikiCqv6R6QIiI4ARAPUKQwtAY4wxrFsH+/dDmzZ5P8eCBZCefqoHqJdI5nZr4u+bxiMpyW23AXFNURfKkrUkwPf/mTrA9gD79gem+a5Q1e2enxuBuUBs1oNUdaKqtlTVluedd15+xGyMMeYM/PkndOwI3bqdPmZVbsyd6way9W2v5tWlC2zb5uYKzY53QNyyZfMehzGFQSiTtcXAJSLSQETK4hKy03p1ikhjoDrwi8+66iJSzvO8JnA5gdu6GWOMKQROnnTtffbudb0wP/8852MCiYtziVr58qdv81aN5lQVasN2mOIiZMmaqqYB9wLfAr8Dn6rqKhF5WkR8e3cOAKarZuqIfRmwRERWAHHA86pqyZoxxhRizz4LP/4Ib78NjRrBpEl5O8/+/bBs2elVoF6NG8MFF+ScrNmAuKa4COncoKo6G5idZd1TWZbH+TnuZyAylLEZY4zJP/PmwbhxMGgQDBvmqkPHjHFDbDRokLtzLVjg2qNl7Vzg5W23Nndu9u3WkpLg6qtzd21jCiObwcAYY8wZ2b0bBgyAiy+GN990ydPgwe7n++/n/nxxca76M7sOCl26wI4drjODP8nJbgYEK1kLralTITzcDVgcHu6WTf6zZM0YY0yepafD7bfDvn3w6adQpYpbX6cOdO8Okye7tmy5ERfnJm4vVy7wPr7jrfljw3aE3tSpMGIEbN7sSjg3b3bLlrDlP0vWjDHG5NmECfDf/8LLL0N0dOZtw4a5pOnHH/0f68++fRAfH7gK1Ovii92QHIHarXlnL7CStdB54onTe/ymprr1Jn9ZsmaMMSZPfv7ZfTHfcgvceefp26+/Hs49N3cdDebNy769mpeI28fbbi0rK1k7M8FUb27Z4v/YQOtN3lmyZowxJtf27oX+/U/Nn+ivkX+5cq7DwRdfuBKzYMTFuemlWrXKed8uXWDnTliz5vRtNiBu3gVbvRloLPrCOEZ9UW9bZ8maMcaYXFGFoUNdj89PPoFq1QLvO2wYHDsG06YF3sfX3Llw+eXBDWTrHdrDX1Xo1q1ueI8yZYK7rjkl2OrN8eMzz9sKbnn8+DO7fn4nVsWhbZ0la8YYY3LllVfg66/hhRegZcvs942OhubNg6sK3b0bEhJyrgL1atjQVXP6S9ZsQNy8C7Z6c+BAV6pav74rxfSWsgaaCD2YJCwUiVVxaFtnyZoxxpigLV4MDz8MvXvDqFHBHTN0KCxdCsuXZ7/fvHnuZ6DBcLPKrt2aDYibd7mp3hw4EBITXa/gxMTsE7VgkrBQJFbFoW2dJWvGGGOCcuAA9Ovn2oFNmpT9JOq+br3VVWtOnpz9fnPnQqVKOZfW+erSBfbsgVWrTq1TdcmalazlTSiqN4NNwkKRWBWltnWBWLJmjDEmKHfe6ZKg6dOhevXgjzv3XOjTBz76yLVfCyQuzk0Cn5t2Zv7arSUnw+HDVrKWV7mt3gxGsElYKBKrULWtO5ssWTPGGJOjPXvcoLcPPQRt2+b++GHDXI/QWbP8b9+1C1avDr4K1Cs83D18B8e1YTvOXLDVm8EKNgkLRWIViuTzbLNkzRhjTI4SEtzPYBv/Z3XllS55CtTRwJts5eX83nZr6elu2QbELXyCTcJClVjld/J5tlmyZowxJkfx8e5nZGTeji9VCoYMgW+/PVXy5Ssuzk1V1bx57s/dubMrtfMmlFayVvjkJgkr6olVKFiyZowxJkcJCXDeeVCrVt7PMWSIa/z/4Yenb5s717VXK1069+fNOk9oUpIbHuLCC/MYqAkJS8LyzpI1Y4wxOYqPh6io4HuA+tOwoUusJk06VWUJsGOHm4Ugr1WsdetCo0anOhnYgLimuLFkzRhjTLZOnoSVK/NeBepr2DDYuBEWLDi17kzaq3l16eLGaTt50gbENcWPJWvGGGOy9ccfcOSIK1k7U337QtWqmTsazJ3rpqyKicn7eTt3duPAxcfbgLim+LFkzRhjTLa8DffzI1mrWBEGDIAZM+DgQbcuLg46dXKdEPLKWyoXF2cD4prix5I1Y4wx2YqPdw32mzbNn/MNHepK6j75BLZtg/Xrz6wKFNysCpdeCp9/7kbGt5I1U5zkod+NMcaYkiQ+Hi65BCpUyJ/ztW7tEr9Jk06NvZXbwXD96dzZDQcBVrJmihcrWTPGGJOthIT8qQL1EnEdDRYtgrfeclNXRUef+Xl9S+csWTPFiSVrxhhjAjp0yHUwyM9kDWDQIDem2k8/wRVXuGrWM+VbOmfVoKY4sWTNGGNMQCtXup/5MWyHr1q1oFcv9zw/qkDBja3WpIkNiJudqVPdXKphYe7n1KkFHZEJhiVrxhhjAsrPnqBZjRwJ5cpBz575d86+fV2Val5mQijupk6FESNg82Y3k8TmzW7ZErbCT1S1oGPIFy1bttQlS5YUdBjGGFOsjBoFH3zgxjDLj6rKrI4fh7Jl8+983pkRQhFrURce7hK0rOrXd9M/mbNLRH5T1ZbB7GsfZ2OMMQHFx0NEROiSn/xM1MDFWdIStWCrNrdsyd16U3iUsI+0McaYYKmemhPUFE65qdqsV8//OQKtN4WHJWvGGGP82rbNVX9aslZ4PfGEGwTYV2qqW5/V+PGnxrXzqljRrTeFmyVrxhhj/IqPdz8tWSu8clO1OXCgGzS4fn031l39+m554MDQxmjOnPWXMcYY45c3WYuIKNg4TGD16vnvNBCoanPgQEvOiiIrWTPGGONXQoL70j/nnIKOxARiVZslgyVrxhhj/LLOBYWfVW2WDFYNaowx5jTHj8OaNXDddQUdicmJVW0Wf1ayZowx5jRr1kBampWsGVMYWLJmjDHmNNYTNHdszk0TSlYNaowx5jTx8W52gUsuKehICj/vwLTe8c68A9OCVU+a/GEla8YYU4R88gns3Rv668THQ9OmUKZM6K9V1OVmYFpj8sKSNWOMKSI2b4b+/WHcuNBfKyHBqkCDZXNumlDLMVkTkXtFpPrZCMYYY0xg3nZkU6fC0aOhu86ePbB9O0RGhu4aRUUwbdFszk0TasGUrF0ALBaRT0Wkh4hIqIMyxhhzuoQE93P/fvjqq9Bfp6SXrAU7SboNTGtCLcdkTVXHAJcA7wFDgPUi8qyINApxbMYYY3wkJEDdum7g00mTQnsdsGQt2LZoNjCtCbWg2qypqgJ/eh5pQHVgpoj8K4SxGWOM8ZGQANHRMGQIfPdd6NpExcfDeedBrVqhOX9RkdtJ0hMTIT3d/cyPRM2GAzFewbRZ+6uI/Ab8C/gJiFTVu4EWwI0hjs8YYwxuRoG1a107siFDXLXcBx+E5lrx8e46Jb3RS6jaogWThAVbBWtKhmBK1moCfVW1u6rOUNUTAKqaDvQKaXTGGGOAUzMKREa6L/grr4TJk11JTn46eRJWrrQqUAhNW7RgkzAbDsT4CiZZmw3s8y6ISBURaQOgqr+HKjBjjDGneNuReXtoDhsGmzbBvHn5e52NG+HIEUvWIDRt0YJNwmw4EOMrmGTtLeCQz/JhzzpjjDFnSUKCG6C2cWO33KcPVKuW/x0NvMOD2LAdTn63RQs2CbPhQIyvYJI18XQwADKqP22aKmOMOYsSEqBJk1MzClSoALfeCjNnQnJy/l0nPt61pWraNP/OaU4JNgmz4UCMr2CStY2eTgZlPI/7gI2hDswYY8wp3kb/voYOdYPjfvJJ/l0nIcHNB5o1UTD5I9gkzIYDMb6CSdbuAtoD24AkoA0wIpRBGWOMOWX/fkhKOj1Za9kSIiLytyrUX1Jo8k9ukrBQDAdiiqYcqzNVdRfQ/yzEYowxxo+VK93PrI3+RVxHg/vvh1WroFmzM7vOoUPwxx8wePCZncdkb+BAS7xM7gQzzlp5ERkpIm+KyCTv42wEZ4wx5vSeoL4GDYLSpd0wHmdq1Sr303qCGlO4BFMNOgU3P2h3YB5QB0gJZVDGGGNOSUhwPT/r1Dl923nnwfXXw4cfwokTZ3Ydb09QS9aMKVyCSdYuVtUngcOq+gFwLWAtGowx5ixJSMh+RoFhw2D3bvjPf87sOvHxULmya0dljCk8gknWvP+rHRCRCKAaEB7MyUWkh4isFZENIvKon+0vichyz2OdiBzw2TZYRNZ7HtaCwhhTIqm6NmvZNfrv3h0uvPDMOxp4OxeEBTVrtDHmbAnmV3KiiFQHxgCzgNXAP3M6SERKAW8APYGmwAARyTRyj6qOVtUYVY0BXgM+9xx7LjAW1/O0NTDWE4MxxpQoW7e6cdSyS9ZKl3adAmbPhuh6lN0AACAASURBVB078nYdVVeCZ1WgxhQ+2SZrIhIGHFTV/ao6X1Ubqur5qvp2EOduDWxQ1Y2qehyYDtyQzf4DgGme592B71R1n6ruB74DegRxTWOMKVay61zga+hQN6/nlCl5u862bW6IEBu2w5jCJ9tkzTNbwb15PHdtYKvPcpJn3WlEpD7QAPgxt8caY0xx5k3WIiKy3+/SS6FDB1cVemrOmeBZ5wJjCq9gqkG/E5EHRaSuiJzrfQRxnL+msIH+hPQHZqrqydwcKyIjRGSJiCzZvXt3ECEZY0zREh8PdevCOefkvO+wYbB2LfzyS+6vE2wJnjHm7AsmWRsGjATmA795HkuCOC4JqOuzXAfYHmDf/pyqAg36WFWdqKotVbXleeedF0RIxhhTtOSmHdnNN0OlSnnraJCbpNAYc3blmKypagM/j4ZBnHsxcImINBCRsriEbFbWnUSkMVAd8P1f8Fugm4hU93Qs6OZZZ4wxJcbx47BmTfClXZUrQ79+bq7QQ4dyd634eKsCNaawCmYGg9v9PXI6TlXTcO3dvgV+Bz5V1VUi8rSIXO+z6wBguuqpVhaqug/4By7hWww87VlnjDElxtq1kJaWu6rJYcNcojZzZvDHeJNCS9aMKZxynBsUaOXzvDxwJbAU+DCnA1V1NjA7y7qnsiyPC3DsJMCmtTLGlFh5aUfWvr3rbDBpEgwZEtwxa9a4pNCSNWMKp2Amch/luywi1XBTUBljjAmhhAQ3hlrjxsEfI+KG8XjsMVi3ziVuOfH2BLXOBcYUTnkZpzoVuCS/AzHGGJNZQgI0aQJly+buuNtvd7MQvPxycPOFxse7awST2Bljzr5g2qx9LSKzPI9vgLXAV6EPzRhjSjbvnKC5ddFFcMst8NZbULs23H8/rFiR/XWaNoUyZfIeqzEmdIIpWXsBeNHzeA7opKqnzfNpjDEm/yQnw5Ytea+a/PBD+Ppr6NQJXn8dYmIgNtaVtu3alXlf75ygxpjCKZhkbQvwP1Wdp6o/AXtFJDykURljTAm3cqX7mdckqkwZ6NXL9QrdscMlbKVLw+jRrrStd2/44gu3bft261xgTGEWTLI2A0j3WT7pWWeMMSZE8rPRf40aMHIkLF7sksDRo+F//4O+feHii90+hTlZmzoVwsNdO7zwcLdsTEkSTLJW2jMROwCe57ls7mqMMSY3EhKgalWoVy9/z9usGfzrX7B1K8ye7UrfIiKgdev8vU5+mToVRoyAzZvdnKebN7tlS9hMSRJMsrbbdxBbEbkB2BO6kIwxxng7F4i/mZLzQenS0LOnm+0gIaHwTjP1xBOQmpp5XWqqW29MSRFMsnYX8LiIbBGRLcAjwJ2hDcsYY0ou1bz3BC1utmzJ3XpjiqNgBsX9A2grIpUBUdWU0IdljDElV1KS6w1qyZqrBt682f96Y0qKYMZZe1ZEzlHVQ6qa4plc/ZmzEZwxxpREeZlmqrgaPx4qVsy8rmJFt96YkiKYatCeqnrAu6Cq+4FrQheSMcaUbN5kLSKiYOMoDAYOhIkToX59136vfn23PHBgQUdmzNkTzETupUSknKoeAxCRCkC50IZljDElV0IC1KkD1asXdCSFw8CBlpyZki2YkrWPgB9E5C8i8hfgO+CD0IZljDEll3UuyBsbj80UV8F0MPiXiMQDVwEC/BeoH+rAjDGmJDpxAn7/HXr0KOhIihbveGzeYT6847GBlcqZoi+YkjWAP3GzGNwIXAn8HrKIjDGmBFu71iVshXlGgcLIxmMzxVnAZE1ELhWRp0Tkd+B1YCtu6I4uqvr6WYvQGGNKEOsJmje5HY/NqkxNUZJdNegaYAFwnapuABCR0WclKmOMKaESEtzsAk2aFHQkRUtuxmOzKlNT1GRXDXojrvozTkTeEZErcW3WjDHGhEhCAjRuDGVtBuZcyc14bFZlaoqagMmaqn6hqv2AJsBcYDRQS0TeEpFuZyk+Y4z5/+3deXzU9bX/8dchiIpWRcENJbhAXer204J7rViL0qr3ti40vWpdqF5cqIiiaKVaFPW2Yiu1onJdwL1Wa6/7VnurolBRAWWRHREVsYAoEHLuH2fyI8EkzCQz8/1O5v18POYxmW++y2fyhcnJZzmnrGglaPPkko9NJayk1Kx3gYG7f+HuY939B8AOwERgcMFbJiJSZpYujSE5BWvNU1UFs2dDTU08Nzak2VipKpWwkrTKdjUoAO7+mbvf5u5HFqpBIiLlatKkeFawVlgqYSWlJqdgTURECkcrQYtDJayk1GRTbkpERIrg3XfhG9+I4EEKSyWspJSoZ01EJCXefTeKt1sZrLtXnjOR7ClYExFJAXd4553yqFxQm+dszpx437V5zhSwiTRMwZqISAosWACff14e89WU50wkNwrWRERSoJwWFyjPmUhuFKyJiKRAOQVrynMmkhsFayIiKfDuu9C5M3TokHRLCk95zkRyo2BNRCQFyqnMlPKcieRGedZERBK2ejW89x4cXUZVl5XnTCR76lkTEUnY9OmwalX59KyJSG4UrImIJGTePHjoIRg6NF63hmBNyW5F8k/DoCIiRbByJbz1Frz2WjxefTVyqwFstBH06QN77plsG1uqNtltbQ612mS3oCFPkZYwd0+6DXlxwAEH+Pjx45Nuhoi0Uu6wcCFMmwYzZ8awZTbHfPBBBGcTJkTABjGh/qCD4nHwwbDPPrDBBoVtfzF07RoB2roqK2H27GK3RiTdzGyCux+Qzb7qWRMRqWPp0gjIpk2DqVPXfj1tGixfnvv52rWD/feH/v0jMDvoINh++/y3Ow2U7FakMBSsiYgAt98Ov/wlfPTR2m1m0VvUvTscemg8d+8Ou+4KG2+c3Xk7dIANNyxIk1OnS5eGe9aU7FakZRSsiUjZu+kmuOgiOOwwGDBgbVC2yy4xn0yyM2xY/TlroGS3IvmgYE1Eytrw4XDZZfCjH8F998WwpTRP7SKCIUNi6LNLlwjUtLhApGUUrIlIWXKHq6+OtBl9+8I990BbfSK2mJLdiuSf8qyJSNlxj96foUPh9NPh3nsVqK2P8qeJJEcfTyJSVtxh4MCYp9avH9x6awQg0jjlTxNJlj6iRKRs1NTAeedFoHb++fDHPypQy8aQIfUXDUC8HjIkmfaIlBt9TIlIWVizBn7+c/jDH+Dii+HmmyM1h6yf8qeJJEvBmoi0etXV8LOfwR13wBVXwA03KFDLRWN50pQ/TaQ4FKyJSKu2ejX89KexiOCaa+KhQC1ku2hg2LDIl1aX8qeJFI8WGIhISVq5Ej75BBYvrv/49NP6r2fOhPffj960QYOSbnV65LJoQPnTRJKlQu4iUnDz5sEmm8CWW7b8XF98EYHCb37TeDH1TTeFrbZa++jbN4ZBZS0VXRdJlgq5i0hqLFoEe+8d88Yuuigem2+e+3nc4ZFH4vj586NX5/DDIxjr2HFtYLblluVTi7MltGhApHQoWBORgho0KHrDjjkmKgbccgsMHgz9+399HlRj3n8/Um08/zzssw888AAcckhh293aqei6SOnQAgMRKZi//S0m9g8aBI8/DuPHQ48ecMklsOuukZC2saFMgGXLYt+99oI334Tf/z7OoUCt5bRoQKR0KFgTkYJYtQr+8z9jDlRt8tT994ennoogbpdd4vu77RZ1OdesWXuse/Se7bYb3HgjnHoqTJsWCW1VFio/qqpg1Ki4P2bxPGqUFg2IpFFBgzUz621mU81shpkNbmSfk8xsiplNNrP76mxfY2YTM4+/FLKdIpJ/I0bAlCnRG7ZuD87hh8Mrr0TgtsUWcNppMa/t0Udh8mTo1SsWBWy7Lbz2Gtx5J2y9dTLvozWrqorFBDU18axATSSdCrYa1MwqgGnA94D5wJtAX3efUmefbsBDwJHuvsTMtnb3jzPfW+7um2Z7Pa0GFUmPefOiV+yoo2L4syk1NRGkXXllzE0D6NABrr0Wzj4bKioK314RkWJLy2rQHsAMd5+ZadQDwPHAlDr7nA2MdPclALWBmoiUtgEDYijz5pvXv2+bNvDjH8MJJ0Tur1mzYrizY8fCt1NEpBQUMljrDMyr83o+0HOdfboDmNk/gApgqLs/nfneRmY2HqgGhrv7Y+tewMz6Af0AumgJk0gqPPlk9JRde23k8spW27YxHCoiIvUVMlhrqKDLumOubYFuwBHADsDfzexb7v450MXdPzSznYEXzexdd/+g3sncRwGjIIZB8/0GRCQ3X34ZKTa++U0YODDp1oiItA6FDNbmAzvWeb0D8GED+7zu7quBWWY2lQje3nT3DwHcfaaZvQzsB3yAiOSspgamT4d//Su7/XfYAbbfPvfrDB8e5Z2efx7atcv9+GyNHavSRyJSPgoZrL0JdDOznYAFwCnAT9bZ5zGgL3CXmXUkhkVnmlkHYIW7r8xsPwS4oYBtFWlVli6FN96AV1+N1ZSvvw6ff5798e3aRQLbgQOzT5UxfTpcf32s4uzVq3ntzkYuNS1FRFqDgtYGNbNjgRHEfLTR7j7MzK4Gxrv7X8zMgN8AvYE1wDB3f8DMDgZuA2qI9CIj3P3Opq6l1aBSrtxhxoy1gdmrr8KkSbHdDPbcEw46KB7bbJPd+UaPjnlnPXvC3XfHsOb6jjnmmLj21Kmw3Xb5eW8NaW01LdVLKFKeclkNqkLuIo0YOhTGjYMrrshfxvwvvoicY9bQjM5m+Oc/4Qc/gIUL4/Xmm8OBB64Nznr2bH4dzgcfjJJQK1ZEAHHhhY2n0XjkETjxxFj9ecEFzX8/2WjTJtq3LrMY7i0l6/YSQvz7UHJakdYvl2BNFQxEGvDppzH/6pln4NBDIyCaOLH555s4MbLwb7EFXHZZftpYUwPnnBOBy+23R2/aZ5/B00/DVVfB0Uc3L1CDCHxOOSUS1B59dAyHfuc7MdS5rmXLIlXHvvtGRYJCa2zhdykuCB8ypH6gBvG6tuLDusaOjZ7FNm3ieezYQrdQRNJAwZpIA0aPhpUrY1hx+PAY3ttvPzj55Bjmy0ZNTaSx6NUrjn30UfjWt+C//gvefbflbbznnqiXef31cNZZMdzZJs//o7fdFh57LOp7Tp4cRdR/97v6PVi/+hUsWBB1PotRCqoQNS2TCoLmzs1+e20v3Jw5EaDXztVTwCZSBty9VTz2339/F8mH6mr3rl3dDz987bYlS9yvuMJ9k03c27RxP+MM99mzGz7+yy/dR41y3313d3Dv3Nn9+uvjHJ9+6r7VVu6HHupeU9P8Nv7rX+7bbOPes6f7mjXNP08uFixwP/bYeE+HH+7+wQfu77zjXlHhftZZxWlDrTFj3Csr3c3iecyYlp2rfft4X7WP9u1bds5sVVbWv27to7KyZfuKSPoR8/ezinESD7Ly9VCwJvnyxBPxP+PBB7/+vUWL3AcMcG/XLh4XXOD+0Udrvzd0qHunTnH8fvvFL/yVK+uf48474/t33dX8Ng4aFOcYN67552iOmhr3//5v9802i8C1e/cIPj/9tLjtyKckg6BcAkWzhttpVvh2ikj+KVgTaYHevd2328591arG95kzJ3qTKiril+vxx7tvuGH8j+rTx/3FFxvvOVuzxv2ggyKoW7w49/ZNneq+wQbup5+e+7H5Mneu+9FHx/u9887k2pEPhQqCsu39y3Y/9ayJtC4K1kSaafr0+F8xdGh2+0+b5t63r3uHDu79+rlPmZLdcRMnxnDqOefk3sY+fdy/8Q33hQtzPzafamoicCx1hQiCCjG0muRwrYjkXy7BmhYYlIhly2DVqqRb0frVTpKvTbK6Pt26wX33xSrM226D3XfP7rh99okUF7fdFosEsvXUU/A//wNXXhmT/5NkBt27J9uGfCjEgoVcV3lmo6oqUnpUVsbPvrJSKT5EyoXyrJWA6ur4pdijBzzwQNKtab1WrIDOnSNVxYMPFv56S5fCbrtFWadx4xrPYVZr1SrYa6/oU5k0qbDlnFqLbBPO5jsxbWvKBScihaE8a63M44/DrFkRQLzxRtKtab3uvz9KMvXvX5zrbbYZ3HQTTJgQPWzr8/vfw7RpcYwCtfXLJdVFVVVUP6ipieeW9la1plxwIpI8BWsl4JZb4kO+UycYPLjhv9ilZdzj5/ytb8FhhxXvuiedBEcdBZdfDosWNb7fokVRq/OYY6BPn+K1r5QVYigSssvJVoihVREpXwrWUm7yZHj55cgMf+WV8NJL8OyzSbeq9Xnttagy0L9//kpBZcMsgsQVK+CSSxrf7/LLY5+bbipe20pdLglns5Vtb53ml4lIPmnOWsr17w933gnz58ew2W67RQmhCRPyn62+nFVVwV//Gpn4N920+Ne/4orodXn55SjrVNf48TFf8aKLovqBZKcQBd9bWxF5EUmO5qy1EkuXRkmhk0+Gjh1jntI110QPkBYa5M+iRfDww3D66ckEahA9Z127Rg/q6tVrt7vHqtFOnaJnVbJXiKHIQvTWiYisj4K1FLv3Xli+vP6E9759I+3DlVcqlUe+3HFHBEjFKELemPbto+bmlCkwYsTa7WPHxhDtddc1vyh7LlpTofBCDEVq4YCIJEHDoCnlHoW5N9nk63m4nnoKjj02Vgeed17h2zJxYgzB7rxz4a9VbNXVsNNOMbz83HOFvVY26SGOPx6efx7efx86dIBvfnNtao9CD3vXzseqOym/fXvNtapLPyMRyRcNg7YCL78M773XcBqJ3r1jXtM110TPWyHNnw+HHgq9esGXXxb2Wkl44ol4j4VO15HtxPSbb47v/+IXcO218OGHEZQXY35ioVZPtiZaOCAiSVDPWkr9+Mex8nP+fNh4469/f9w4OPBA+NWv4Je/LFw7Tjwx8rytXh3B4RVXFO5aSTjqqMhdNnNmVC4olFwmpl93Xcxhq6iAn/wk5i0WgxK5iogUj3rWStz8+fDYY3DmmQ0HagA9e8K//zvceCN88klh2vHMM/DII3DVVXGt666LtrVUWuZFvfcevPACnHNOYQM1yG1i+sCBMSy70UYwfHhh21WX5mOJiKSTgrUUuu226Mk499ym9xs2LIapCpFo86uvYj5c9+5w8cWRMmLNmqZzgWUjl6zyhfaHP8QK27POKvy1cgmE2rWDF1+E11+P+WrFokSuIiLppGAtZVatgttvjyz1O+3U9L677QZnnBFBx6xZ+W3HDTfAjBkwciRsuGG0ZdCgKMn0v//b/POmZV7UsmVw991RQWDrrQt/vVwDoe22i2oKxZTrfKy09JCKiLR2mrOWMvffH/OUnnoqFhKsz4IFsOuuMcft3nvz04YPPoiVqCecUD+f2xdfRIDYqVOsUF1f4fGGpGVe1K23RqqO116LuX/FkO9i4UnSqkgRkZbJZc6agrWUOfTQSNI6dWr2KwAvvTTmrk2cCHvv3bLru0dakH/8I9JHrDsMVxtMjhoFZ5+d+/nTkAHeHfbaK3oMx48vbnmp1iIN91FEpJRpgUGJevvtCJLOPTe3VA2DB0fC1Msua3kb/vxnePrpKBre0HypU06JgPLyy+Hzz3M/fxrmRb3yStRcbaoOqIb4mqZM/iIixaNgLUVGjozVnz/7WW7HdegQgdqTT0Yg0lzLl8OAAdE711iyXbPItL94caQNyVXtvKgdd0wuT9XIkfEzO+WUhr+fpkUQaaWVoyIixaNgLSWWLIExY2KIsUOH3I8//3zo3DmGRJs7sn3NNTBvXsznaiqVxX77xQrKW26J9Be5cI9h3gULYiHF7NnFC9RWrYo0JH/6U6RFWbeHr1ZaFkGkWRp6SEVEyoWCtZS4666oENDcTPobbwxDh0a6h8cey/34yZPht7+N1aUHH7z+/YcNi1JYAwZEAJbNsOFXX0Wv4cCBa4vS1y1aXkhvvw09esTw7k9/2nQiYQ3xrZ8y+YuIFI8WGKRATU3UgOzUCV59tfnnqa6OifNffRXFyXv1yu44dzjiCJg0KRY2dOyY3XEjRkRZpF/8InLDNbUy8KOP4N/+LYLJq66CAw6AH/4wgtTTTsvlXeZm9epILHv11bDVVtGm445r+hhNnhcRkULTAoMS89xzkdOspUXZ27aNocXq6iij1KtXBEfrM2ZMzHUbPjz7QA2iF3D33aN2ZVPDhhMmwLe/De+8Aw8/HD2AffrAPvtEVYQ1a7K/Zi4mTYq0HL/8ZeRTmzx5/YEalM4QnxZBiIiUCXdvFY/999/fS9UPf+i+9dbuX32Vn/N9+aX7iBHunTq5g/txx7m/807D+y5ZEtfu2dN9zZrcr/XMM3GNhh5m7vff777RRu477uj+1lv1j33wwdjv4Ydzv25TVq92v/Za93bt4mfwpz/lfo4xY9wrK+M9VFbG6zQZM8a9ffv6P+/27dPXTklG2v/9iog7MN6zjHESD7Ly9SjVYG3WrPhAHTIk/+detsz9179233zzuMZPfuI+fXr9ffr3d2/Txv2f/2z+dTbeuOFgbbPN4vmQQ9w/+ujrx1VXu3fv7r7ffu41Nc2/fl1Tprj36BHXPfFE948/zs95iymbX7SVlQ3/zCsri9vWbChwKC4F8iKlQcFakVRXu8+c6f700+433xyBz/e+F7+QttrK/cAD3U891X3YsOg9evtt9y++qH+OSy+NYGnu3MK1c/Fi98suiw/sigr3fv3c581zHz8+foGef37Lzv+b33w9aKioiOczz2y6x3D06NjvySdb1obqavcbb3TfcMP42T/4YMvOl5Rsf9GaNRysmSXT7sYocCi+UgrkRcpZLsGaFhhkqaYmaklOnQrTpsVjxgxYuXLtPpttFgsFunePlZLTp8d+CxbUP1eXLrFP9+5Rzuk734FHHy1Y0/+/jz6Ca6+FP/4x5jl16hQT8KdOjaS6LXHccfDEE/H1BhvEPLQRI2IeXlMVAlatinJZlZXw9783//qXXhr1TE84Id7fNts0/1xJynZxQ6ksgiiVdrYmaSnpJiJNU7mpAunYEZYujeCie/e1gVntY+utGw5Mli9fG7jVPqZOjceyZfC3v8FhhxW06fXMmROrI++9NxYXnHRSy8+5bFn8PBYuhC22iIUERx2V3bG33BJ54l55pXk/h5deisUUZ58dgVopl4/K9hdtrrU5k6pLqsCh+BQgi5SGXIK1xIcv8/UoxjDo/PkxeT1famrcV6zI3/lytWpVfs/3xBMxDDxtWm7HrVgRixy+//3cr/nZZ+477BBz35Yvz/34tMllCCvbuWBJDkUW4v1I0zT0LFIa0Jw1KTXXXRf/GsePz/6Ymhr3k092b9vW/c03C9e2YirEL9ok5zBl+34UYOSXAl+R9MslWNMwqKTC0qUxPNerV5SDysaYMfAf/xFDepdfXtj2FVO+hyyTHorM5v1o6E5Eyo3mrElJuvJK+PWvI3ntHns0ve/s2ZFUd++94eWXoaKiGC0sTaUQCCUdUIqIFJsqGEhJuvDCmCQ/fHjT+61ZA6eeGr/c771Xgdr6lEJFhi5dctsuIlJOFKxJanTsCD//Odx3H8ya1fh+N9wQaT5GjoxeI2laKRRdL4WAUkQkKRoGlVRZsAB23hnOOANuvfXr3x8/Hg46CH70I7j//tJO0yH1JZVeREQkCRoGlZLVuTOcfjqMHh052+r64ov45b3tthHINSdQU/Hz9Kqqijl0NTXxrEBNRCQoWJPUueQSqK6G3/62/vaLL47kwvfcAx06rN2ebQBWm0h2zpyY7zZnTrxWwCYiImmmYK0ElFtv0C67QN++0Xu2eHFs++tfozrBwIHw3e+u3TeXAGzIkPoZ/yFeDxnS/LaW270REZHi05y1lMu1rFBrMWkS7LUXXHUVnHtufL399jBuHGy44dr9cklLke/0EOV6b0REpOWUZ60VKYUcWYVywglRL/Tb3476qRMmwJ571t8nlwAs3z/Lcr43IiLSMlpg0IrMnZvb9tbk8sthyRJ49tlI17FuoAa55efKd3qIcr43IiJSPArWUq6ck4X26BHDiX37wnnnNbxPLgFYvvONlfO9ERGR4lGwlnLlnix0zJhIktumkX+puQZg+UwPUe73RkREikPBWsqVQvb5pCWVn0v3RkREikELDERERESKTAsMRERERFoJBWsiIiIiKaZgTURERCTFChqsmVlvM5tqZjPMbHAj+5xkZlPMbLKZ3Vdn+2lmNj3zOK2Q7RQRERFJq4IFa2ZWAYwEjgH2APqa2R7r7NMNuAw4xN33BAZktm8JXAX0BHoAV5lZB0qAakWKiIhIPhWyZ60HMMPdZ7r7KuAB4Ph19jkbGOnuSwDc/ePM9u8Dz7n7Z5nvPQf0LmBb8yKXouKFur4CRRERkdalkMFaZ2BendfzM9vq6g50N7N/mNnrZtY7h2Mxs35mNt7Mxn/yySd5bHrzDBlSv6g3xOshQwp/7UIFigoARUREklXIYM0a2LZuUre2QDfgCKAvcIeZbZHlsbj7KHc/wN0P6NSpUwub23JJ1oosRKCYdE+hiIiIFDZYmw/sWOf1DsCHDezzuLuvdvdZwFQieMvm2NRJslZkroFiNj1mSfYUioiISChksPYm0M3MdjKzdsApwF/W2ecx4LsAZtaRGBadCTwDHG1mHTILC47ObEu1JGtF5hIoZttjlmRPoYiIiISCBWvuXg2cRwRZ7wEPuftkM7vazI7L7PYMsNjMpgAvAYPcfbG7fwZcQwR8bwJXZ7alWpK1InMJFLPtMUuyp1BERESCaoO2ImPHRsA1d24EVMOGNRwotmkTPWrrMoti6HXP169f/cCufXsVKxcREWkp1QYtEfleaVlVBbNnR8A1e3bjAVW2PWZJ9hSKiIhIULCWkCRXWuYyZJptACgiIiKFoWAtIUmutFSPmYiISOnQnLWEZDtvTERERFofzVkrAVppKSIiItlQsJaQJHOyiYiISOlQsJYQzRsTERGRbLRNugHlrKpKwZmIiIg0TT1rIiIiIimmYE1EREQkxRSsiYiIiKSYgjURERGRFFOwJiIiZBkkAgAABvRJREFUIpJiCtZEREREUkzBmoiIiEiKKVgTERERSTEFayIiIiIppmAtS2PHQteu0KZNPI8dm3SLREREpBwoWMvC2LHQrx/MmQPu8dyvnwK2QlOALCIiomAtK0OGwIoV9betWBHbpTAUIIuIiAQFa1mYOze37dJyCpBFRESCgrUsdOmS23ZpOQXIIiIiQcFaFoYNg/bt629r3z62S2EoQBYREQkK1rJQVQWjRkFlJZjF86hRsV0KQwGyiIhIaJt0A0pFVZWCs2Kq/VkPGRJDn126RKCmeyAiIuVGwZqklgJkERERDYOKiIiIpJqCNREREZEUU7AmIiIikmIK1kRERERSTMGaiIiISIopWBMRERFJMQVrIiIiIimmYE1EREQkxRSsiYiIiKSYgjURERGRFDN3T7oNeWFmnwBzmnl4R+DTPDZH8kv3J910f9JL9ybddH/Sqxj3ptLdO2WzY6sJ1lrCzMa7+wFJt0MapvuTbro/6aV7k266P+mVtnujYVARERGRFFOwJiIiIpJiCtbCqKQbIE3S/Uk33Z/00r1JN92f9ErVvdGcNREREZEUU8+aiIiISIqVfbBmZr3NbKqZzTCzwUm3p9yZ2Wgz+9jMJtXZtqWZPWdm0zPPHZJsY7kysx3N7CUze8/MJpvZhZntuj8pYGYbmdkbZvZ25v78KrN9JzMbl7k/D5pZu6TbWq7MrMLM3jKzv2Ze696khJnNNrN3zWyimY3PbEvNZ1tZB2tmVgGMBI4B9gD6mtkeybaq7N0F9F5n22DgBXfvBryQeS3FVw0MdPfdgQOB/pn/L7o/6bASONLd9wH2BXqb2YHA9cBNmfuzBDgzwTaWuwuB9+q81r1Jl++6+751Unak5rOtrIM1oAcww91nuvsq4AHg+ITbVNbc/RXgs3U2Hw/cnfn6buCEojZKAHD3he7+z8zXy4hfOp3R/UkFD8szLzfIPBw4Engks133JyFmtgPQB7gj89rQvUm71Hy2lXuw1hmYV+f1/Mw2SZdt3H0hRMAAbJ1we8qemXUF9gPGofuTGplhtonAx8BzwAfA5+5endlFn3HJGQFcAtRkXm+F7k2aOPCsmU0ws36Zban5bGub1IVTwhrYpuWxIk0ws02BPwED3H1pdBBIGrj7GmBfM9sC+DOwe0O7FbdVYmY/AD529wlmdkTt5gZ21b1JziHu/qGZbQ08Z2bvJ92gusq9Z20+sGOd1zsAHybUFmncIjPbDiDz/HHC7SlbZrYBEaiNdfdHM5t1f1LG3T8HXibmFm5hZrV/mOszLhmHAMeZ2Wxius2RRE+b7k1KuPuHmeePiT90epCiz7ZyD9beBLplVuS0A04B/pJwm+Tr/gKclvn6NODxBNtStjJzbO4E3nP339b5lu5PCphZp0yPGma2MXAUMa/wJeDHmd10fxLg7pe5+w7u3pX4PfOiu1ehe5MKZraJmX2j9mvgaGASKfpsK/ukuGZ2LPEXTgUw2t2HJdyksmZm9wNHAB2BRcBVwGPAQ0AXYC5woruvuwhBCszMDgX+DrzL2nk3lxPz1nR/EmZmexOToCuIP8QfcverzWxnojdnS+At4KfuvjK5lpa3zDDoxe7+A92bdMjchz9nXrYF7nP3YWa2FSn5bCv7YE1EREQkzcp9GFREREQk1RSsiYiIiKSYgjURERGRFFOwJiIiIpJiCtZEREREUkzBmoi0ama2xswm1nnkrRizmXU1s0n5Op+ISEPKvdyUiLR+X7r7vkk3QkSkudSzJiJlycxmm9n1ZvZG5rFrZnulmb1gZu9knrtktm9jZn82s7czj4Mzp6ows9vNbLKZPZupHoCZXWBmUzLneSChtykirYCCNRFp7TZeZxj05DrfW+ruPYBbiEomZL6+x933BsYCv8ts/x3wN3ffB/h/wOTM9m7ASHffE/gc+FFm+2Bgv8x5zinUmxOR1k8VDESkVTOz5e6+aQPbZwNHuvvMTIH6j9x9KzP7FNjO3Vdnti90945m9gmwQ91yQGbWFXjO3btlXl8KbODuvzazp4HlRLm0x9x9eYHfqoi0UupZE5Fy5o183dg+Dalby3ENa+cC9wFGAvsDE8xMc4RFpFkUrIlIOTu5zvNrma9fBU7JfF0F/G/m6xeAcwHMrMLMNmvspGbWBtjR3V8CLgG2AL7Wuycikg39pScird3GZjaxzuun3b02fceGZjaO+MO1b2bbBcBoMxsEfAL8LLP9QmCUmZ1J9KCdCyxs5JoVwBgz2xww4CZ3/zxv70hEyormrIlIWcrMWTvA3T9Nui0iIk3RMKiIiIhIiqlnTURERCTF1LMmIiIikmIK1kRERERSTMGaiIiISIopWBMRERFJMQVrIiIiIimmYE1EREQkxf4PsZqt495rTHUAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 720x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(10, 6))  # Grafiği büyütelim\n",
    "train_acc_augmentation = history_augmentation.history['acc']\n",
    "val_acc_augmentation = history_augmentation.history['val_acc']\n",
    "epochs_augmentation = range(1, len(train_acc_augmentation) + 1)\n",
    "\n",
    "plt.plot(epochs_augmentation, train_acc_augmentation, 'bo', label='Training accuracy with Data Augmentation')\n",
    "plt.plot(epochs_augmentation, val_acc_augmentation, 'b', label='Validation accuracy with Data Augmentation')\n",
    "plt.title('Training and validation accuracy with Data Augmentation')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6f836157",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('model_augmentation.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb1925bb",
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
