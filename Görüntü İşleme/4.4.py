{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "768ba0d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "921b3f5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_dir = 'C:\\\\Users\\\\user\\\\source\\\\proje_odev\\\\train'\n",
    "validation_dir = 'C:\\\\Users\\\\user\\\\source\\\\proje_odev\\\\validation'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "7cf4c583",
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
   "execution_count": 12,
   "id": "32a711ed",
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
    "    Dropout(0.40),  # Dropout ekleyelim\n",
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
   "execution_count": 13,
   "id": "4297e244",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "100/100 [==============================] - 8s 78ms/step - loss: 0.6761 - acc: 0.5994 - val_loss: 0.6686 - val_acc: 0.5994\n",
      "Epoch 2/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6752 - acc: 0.5916 - val_loss: 0.6710 - val_acc: 0.5994\n",
      "Epoch 3/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6746 - acc: 0.5941 - val_loss: 0.6693 - val_acc: 0.6175\n",
      "Epoch 4/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6726 - acc: 0.5928 - val_loss: 0.6551 - val_acc: 0.6162\n",
      "Epoch 5/50\n",
      "100/100 [==============================] - 6s 63ms/step - loss: 0.6618 - acc: 0.6075 - val_loss: 0.6479 - val_acc: 0.6219\n",
      "Epoch 6/50\n",
      "100/100 [==============================] - 6s 63ms/step - loss: 0.6583 - acc: 0.6150 - val_loss: 0.6444 - val_acc: 0.6275\n",
      "Epoch 7/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6649 - acc: 0.6006 - val_loss: 0.6521 - val_acc: 0.6375\n",
      "Epoch 8/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6636 - acc: 0.6072 - val_loss: 0.6490 - val_acc: 0.6181\n",
      "Epoch 9/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6581 - acc: 0.6175 - val_loss: 0.6547 - val_acc: 0.6044\n",
      "Epoch 10/50\n",
      "100/100 [==============================] - 6s 64ms/step - loss: 0.6551 - acc: 0.6146 - val_loss: 0.6327 - val_acc: 0.6438\n",
      "Epoch 11/50\n",
      "100/100 [==============================] - 6s 65ms/step - loss: 0.6590 - acc: 0.6116 - val_loss: 0.6360 - val_acc: 0.6362\n",
      "Epoch 12/50\n",
      "100/100 [==============================] - 6s 65ms/step - loss: 0.6519 - acc: 0.6319 - val_loss: 0.6348 - val_acc: 0.6381\n",
      "Epoch 13/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6499 - acc: 0.6344 - val_loss: 0.6324 - val_acc: 0.6425\n",
      "Epoch 14/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6602 - acc: 0.6078 - val_loss: 0.6240 - val_acc: 0.6456\n",
      "Epoch 15/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6472 - acc: 0.6203 - val_loss: 0.6202 - val_acc: 0.6475\n",
      "Epoch 16/50\n",
      "100/100 [==============================] - 7s 65ms/step - loss: 0.6455 - acc: 0.6300 - val_loss: 0.6277 - val_acc: 0.6475\n",
      "Epoch 17/50\n",
      "100/100 [==============================] - 7s 65ms/step - loss: 0.6409 - acc: 0.6260 - val_loss: 0.6189 - val_acc: 0.6544\n",
      "Epoch 18/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.6555 - acc: 0.6091 - val_loss: 0.6277 - val_acc: 0.6650\n",
      "Epoch 19/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6451 - acc: 0.6259 - val_loss: 0.6062 - val_acc: 0.6550\n",
      "Epoch 20/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6402 - acc: 0.6300 - val_loss: 0.6173 - val_acc: 0.6450\n",
      "Epoch 21/50\n",
      "100/100 [==============================] - 7s 66ms/step - loss: 0.6288 - acc: 0.6450 - val_loss: 0.5822 - val_acc: 0.6937\n",
      "Epoch 22/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.6402 - acc: 0.6318 - val_loss: 0.5912 - val_acc: 0.6794\n",
      "Epoch 23/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.6300 - acc: 0.6378 - val_loss: 0.5974 - val_acc: 0.6500\n",
      "Epoch 24/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.6225 - acc: 0.6553 - val_loss: 0.5643 - val_acc: 0.6806\n",
      "Epoch 25/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.6117 - acc: 0.6559 - val_loss: 0.5594 - val_acc: 0.6950\n",
      "Epoch 26/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.6126 - acc: 0.6516 - val_loss: 0.5305 - val_acc: 0.7344\n",
      "Epoch 27/50\n",
      "100/100 [==============================] - 7s 71ms/step - loss: 0.6054 - acc: 0.6581 - val_loss: 0.5928 - val_acc: 0.6831\n",
      "Epoch 28/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5953 - acc: 0.6653 - val_loss: 0.5166 - val_acc: 0.7475\n",
      "Epoch 29/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5954 - acc: 0.6677 - val_loss: 0.5086 - val_acc: 0.7350\n",
      "Epoch 30/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.5767 - acc: 0.6859 - val_loss: 0.5108 - val_acc: 0.7331\n",
      "Epoch 31/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5816 - acc: 0.6766 - val_loss: 0.4917 - val_acc: 0.7350\n",
      "Epoch 32/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.5723 - acc: 0.6825 - val_loss: 0.4789 - val_acc: 0.7594\n",
      "Epoch 33/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.5638 - acc: 0.7035 - val_loss: 0.4803 - val_acc: 0.7512\n",
      "Epoch 34/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5611 - acc: 0.6928 - val_loss: 0.4670 - val_acc: 0.7538\n",
      "Epoch 35/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5633 - acc: 0.6947 - val_loss: 0.4629 - val_acc: 0.7719\n",
      "Epoch 36/50\n",
      "100/100 [==============================] - 7s 67ms/step - loss: 0.5289 - acc: 0.7184 - val_loss: 0.4451 - val_acc: 0.7850\n",
      "Epoch 37/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.5343 - acc: 0.7139 - val_loss: 0.4605 - val_acc: 0.7762\n",
      "Epoch 38/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5410 - acc: 0.7172 - val_loss: 0.4258 - val_acc: 0.7844\n",
      "Epoch 39/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5291 - acc: 0.7184 - val_loss: 0.4361 - val_acc: 0.7863\n",
      "Epoch 40/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5361 - acc: 0.7234 - val_loss: 0.4349 - val_acc: 0.7919\n",
      "Epoch 41/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5175 - acc: 0.7350 - val_loss: 0.4278 - val_acc: 0.7913\n",
      "Epoch 42/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5144 - acc: 0.7322 - val_loss: 0.4262 - val_acc: 0.7925\n",
      "Epoch 43/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5022 - acc: 0.7394 - val_loss: 0.4038 - val_acc: 0.7963\n",
      "Epoch 44/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5324 - acc: 0.7266 - val_loss: 0.4018 - val_acc: 0.8106\n",
      "Epoch 45/50\n",
      "100/100 [==============================] - 7s 68ms/step - loss: 0.5147 - acc: 0.7319 - val_loss: 0.4159 - val_acc: 0.7894\n",
      "Epoch 46/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.5245 - acc: 0.7319 - val_loss: 0.4170 - val_acc: 0.7937\n",
      "Epoch 47/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.5119 - acc: 0.7419 - val_loss: 0.4140 - val_acc: 0.7875\n",
      "Epoch 48/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.5160 - acc: 0.7394 - val_loss: 0.3777 - val_acc: 0.8263\n",
      "Epoch 49/50\n",
      "100/100 [==============================] - 7s 69ms/step - loss: 0.4991 - acc: 0.7399 - val_loss: 0.3760 - val_acc: 0.8150\n",
      "Epoch 50/50\n",
      "100/100 [==============================] - 7s 70ms/step - loss: 0.4974 - acc: 0.7478 - val_loss: 0.3858 - val_acc: 0.8219\n"
     ]
    }
   ],
   "source": [
    "history_combined = model.fit_generator(\n",
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
   "execution_count": 14,
   "id": "45752de7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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1459+/YB8Oijj/Liiy+6t3/wwQdp06YNTZs2ZeXKlQCcOHGCa665hsjISIYNG0ZsbKzXUqsJEybQunVrd3z2foLff/+dyy+/nMjISGJiYkhMTATgmWeeITw8nMjISB555JFMx0pLS6NRo0YAHDhwgJCQEHc87dq1IzExkRkzZnD33XezbNkyvvjiC8aPH09UVJT7+LNnz852Lrm59NJLef7553n55Zfd1+bmm2+mR48ejBo1ipSUFEaMGEF4eDgxMTEsXboUgBkzZnD11VfTq1cvmjZtylNPPeU+5nPPPUdYWBhhYWH85z//cb+HUVFR7m2effZZnnrqKebMmcO6desYMmSI1xLU119/ndatWxMZGcngwYNJSUkBbMnYXXfdRfv27WnUqBELFiwAbMnPuHHjaNGiBf369ePAgQN5XoOQkBDuvfdeqlatytdffw3g9f6aOnUq+/btIy4uju7du+e4nTezZs3innvu4cILLyQhIcG9vG7duhw5cgSAH3/80X3cffv20a1bN2JiYhg3bhwXXXQRR44ccX8WRo8eTcuWLbnhhhv46quvaN++PU2aNHHPSpKUlMTIkSNp06YN0dHRfPrpp+73bdCgQfTq1YvGjRvz0EMPAfDggw9y/PhxoqKiuOGGGwDo168frVq1omXLlsyYMcPrdq7Pmeva33PPPYSFhREeHs68efMA+Oabb+jWrRsDBw6kadOm7uMrpZTf+JrVBfvPuZas1a8vYtO0zD/16/t8iFx5lqxt2bJFjDESHx/vXn/w4EERsf/pd+zYUTZu3CgiZ0ppUlNTBZAvvvhCRETGjx8vkyZNEhGRRx55RKZOnere/v777xcRW/rRq1cvERGZNGmSjBs3TkRE1q1bJyEhIV5Lf1xxZGRkyNChQ92vFxMTI4sWLRIRkZSUFDlx4oQsWrRIOnbsKMnJyZn29dStWzf57bffZMGCBRIbGyvPPvusJCcnS6NGjUREZPr06XLXXXeJiPeSNW/n4ilryZqIyP79+6VChQrua9O6dWtJSUkREZFnn31WxowZIyIiGzZskHr16smpU6dk+vTpUqdOHTl06JAkJSVJ8+bNZe3atbJq1SqJiIiQEydOyLFjx6RZs2ayfv36bK87adIk+de//pXpPfPmwIED7scPPPCAvPrqq+5zHzp0qGRkZMj69euladOmIiIyZ84c6d27t6Snp8tff/0lFStWzLNkzeW2226TKVOmiEjO91fWkrWctvOUlJQktWvXlpSUFHnllVdk/Pjx7nWex/vhhx+kW7duIiJy8803y3PPPSciIp9++qkAcvjwYdmyZYuULFlSNm7cKOnp6RIZGel+f+bNmyfXXHONiIjcd999MmvWLBEROXTokDRu3FhSUlJk+vTpcumll8qxY8ckOTlZ6tatK7t27fJaYuY6txMnTkjz5s3l0KFD2bbzfD579mzp1auXpKWlyZ49e6Ru3bqyd+9e+d///ieVK1eW3bt3S1pamsTGxsoPP/yQ7Tr5SkvWlCqe0JK1/NuxI3/Lz9Ull1xC69at3c9nzZpFTEwMMTExbNq0iV9//TXbPmXLluWKK64AoFWrVu7Sp6wGDhyYbZvly5czdOhQACIjI2nZsqXXfb/99lvatGlDZGQkS5YsYePGjRw+fJgDBw7Qr18/wI4NVa5cOb755htGjx5N2bJlAahatWq248XFxbF06VKWLl3KQw89xLJly1i1ahVt27b14Sp5P5e8iFMa6HLVVVcRGhoK2Otw/fXXA9CyZUvq1KnD1q1bAejVqxdVqlShfPnyDBgwgOXLl7Ns2TKuueYaypUrR8WKFd3Lz9bPP/9MXFwc4eHhzJ49m40bN7rXDRgwAGMMERER7Nq1C4ClS5cybNgwQkJCqFu3Ll26dPH5tTyvgy/3l6/bLVq0iB49ehAaGsrgwYOZP38+GRkZucbief/17duXihUrutddeumltGjRgpCQEFq0aOEujQsPD3e/519//TVPP/00UVFRdO3alZMnT7LD+XB2796dihUrUrZsWZo1a+ZentXUqVPdpdI7d+7kjz/+yDPma6+9lhIlSlCrVi06duzoLum77LLLqF27NiVKlMhUEqyUOr/S0uCXXwIdhf9psuaoVy9/y89V+fLl3Y+3bNnCSy+9xHfffcfPP/9M7969vTb6Ll26tPtxiRIlSEtL83rsMmXKZNsmawLjTXJyMrfffjsLFizg559/ZvTo0e44vPVaE5E8e7PFxcWxbNkyVq9eTd++fTlw4ABLly6lU6dOecaT07nkZe3atTRv3tz93PNa53Ydsp6LMSbH7UuWLJkpQfG1kf4NN9zAa6+9xi+//MKjjz6aaT/XuWaN82x7DK5bt47mzZv7fH/5ut2sWbP48ssvadCgAa1bt2bfvn3u6mTP6+K5b27X3fO8Q0JC3M9DQkIy3b8LFy5k3bp1rFu3jh07dtCkSZNs++d0n3zzzTcsXbqUH3/8kfXr1xMREZHne+ZrzPm5N5VSBWvmTIiIgDlzAh2Jf2my5nj6aShXLvOycuXscn87duwYFStW5IILLmDPnj189dVXBf4aHTt2ZO7cuQD88ssvXktMUlJSCAkJoXr16hw/fpz58+cDUKVKFapXr+5uJ3Ty5EmSk5Pp2bMnb775prvd1aFDh7Ids127dixZsoTSpUtTunRpwsPDmT59utcedxUrVuT48ePndJ7btm3jvvvu44477vC6vlOnTu6es5s2bWLPnj1ceumlgC29OXLkCMnJyXzyySd06NCBTp06sWDBAlJSUkhKSuKTTz4hLi6OWrVqsXv3bg4fPszJkyf5/PPPfTqPEydOUKtWLVJTU/nwww/zPJ9OnToxe/ZsMjIy2LVrF0uWLMlzHxFh6tSpHDx4kB49euR6f3nG6st9ePjwYVatWsXOnTtJTEwkMTGRl19+mVmzZgHQoEED1qxZA+C+fyDz/ffFF1/k+33u1auXux0i2IQ8NyVL2vG+XUnU0aNHqVq1KmXLlmXjxo3udnZZt/Pkuvbp6ens3buXFStWEBsbm6+4lVL+5aroGDsWtm8PbCz+pMmaY/hwmDYN6tcHY+zvadPscn+LiYmhRYsWhIWFcdNNN9GhQ4cCf4077riDXbt2ERERwfPPP09YWBiVKlXKtE21atUYMWIEYWFhXH311ZmqKmfOnMnzzz9PREQEHTt2ZP/+/fTt25fevXsTGxtLVFQUU6dOzfa6ZcuWpU6dOrRv3x6wJW3Jycm0aNEi27bDhg3jmWeeyXe10ubNm4mOjqZZs2YMHTqUf/7zn+6qTm/XISUlhfDwcIYPH857773nLrHs2LEj1157LdHR0QwbNoyoqCjatGnDsGHDaN26NZdddhm33nor4eHhhIaG8vDDD9O6dWv69++f6XxGjRrFmDFjvHYwePLJJ2nTpg09evTweg2yGjRoEPXq1SMsLIzbb7891xLJ8ePHExkZSdOmTVm3bh3fffcdpUqVyvX+Gjt2LN27d6d79+4+3Yfz58+nR48emaZHGjBgAAsWLCA1NZWJEycybtw44uLiMpUEP/HEE3z++efExMTw3XffceGFF2Yq8czLhAkTSE5OJjw8nJYtWzJx4sQ897nxxhuJiIjghhtuoE+fPiQnJxMZGcmTTz6Z6d723M7ToEGDaNasGZGRkXTv3p0XXniBmjVr+hyzUsr/4uMhMtI+vvZaSE0NbDz+YnypHisMYmNjxdWexGXTpk2ZqsOKs7S0NNLS0ggNDWXLli307NmTLVu2uEsWirsZM2awYcMGd69aVbBOnjxJyZIlKVmyJMuXL+fuu+8m6+e1uNLvKaXOzokTcMEF8Oij0KIFDB0KDz98fmrECoIxZo2I+FRcr3+pi4mkpCS6detGWloaIsIbb7yhiZo6bxITExk2bBjp6emUKVOGN954I9AhKaUKuTVrICMD2rSBPn3g669h0iTo1g0uv/zcjp2RASFBVPeof62LicqVK7vbEqnsxowZE+gQirRmzZrl2c5MKaXyIz7e/nYNrPDyy7BiBVx3HaxfDzVq5P+YIvDEE7BlC3zwgW0WFQyCKG9USimllPJNfDw0aACupqTly8Ps2XDwIIwaZROv/EhJgWHDbLJWsqQdFiRYaLKmlFJKqUInPh6yDtkZFQVTpsDnn4Mz4YxP9uyBLl1g7lz497/hnXfAox9VwGmyppRSSqlCZe9eOy1kmzbZ191+O/TrB/fdB760vli3zh5nwwb4+GO4//7gqf500WRNKaWUUoWKa0pib8maMfDWW1C9uu0hmpSU83E++QQ6drSPV6yAAQMKPtaCoMmaH3Xp0iXbwKIvvvgi48aNy3W/ChUqALB7924GDRqU47HzGvrgxRdfJDk52f38yiuvdE+yXZh4Xod169bxxRdfuNdNnDiRKVOm5HmMBg0aEB4eTnh4OC1atODRRx/l1KlTfos5L4sXL/ZpYvqz8c4773D77bd7XV6jRg2io6Np3LgxvXr18imGhQsX5jg9VV6uuuoq2rVrd1b7ng/vvPMOu3fvzvd2Y8aMOetropQ6d/HxUKIEREd7X1+9uu0gsGUL3Hln9vUi8NxzcPXV0LKlPV5UlH9jPhearPnRsGHDmD17dqZls2fPZtiwYT7tX6dOHebNm3fWr581Wfviiy+oXLnyWR8vUDyvQ9ZkLT++//57fvnlF+Lj49m2bRtjx47Ntk16evo5xeorfyZruRkyZAhr165ly5YtPPjggwwcOJBNmzblus/ZJmtHjhzhp59+4siRI2wP0qHFzzZZmzFjhk+DGiul/GPVKggLs50KctK1KzzyCLz9NjiTrABw6hSMHg0PPAD/+AcsXgy1a/s95HOiyZofDRo0iM8++8xdgpOYmMju3bvp2LGje9yzmJgYwsPD+eSTT7Ltn5iYSFhYGGCngho6dCgREREMGTLEPcUTwK233kpsbCwtW7ZkwoQJALz88svs3r2brl270rVrV8CWLh04cACAF154gbCwMMLCwtwDwSYmJtK8eXNuuukmWrZsSc+ePTO9jsunn35K27ZtiY6Opnv37uzduxewY7mNGjWK8PBwIiIi3NMNffnll8TExBAZGUm3bt2yHe/KK6/k559/BiA6Oponn3y1A77RAAAgAElEQVQSgMcee4wZM2a4r8Pp06d5/PHHmTNnDlFRUcxxJoP79ddf6dKlC40aNco0JVFOKlSowOuvv87ChQs5dOgQixcvpmvXrlx77bWEh4fnen2aNWvGiBEjiIiIYNCgQe5k+NtvvyU6Oprw8HBGjx7tfs89r/nq1avp0qULiYmJvP7660ydOpWoqCiWLVuWKb74+Hjat29PdHQ07du3Z/PmzYBNGAYOHEjv3r1p3Lgx999/v3uft99+myZNmtC5c2dWrFiR5zUA6Nq1K2PHjmXatGkATJ8+ndatWxMZGck111xDcnIyK1euZNGiRdx3331ERUXxxx9/eN3Om/nz59OvXz+GDh2a6Z+WkSNHZvonxFWSnJGRwbhx42jZsiV9+/blyiuvdG/XoEEDHn74Ydq1a0dsbCw//fQTvXr14pJLLuH11193H2vy5Mm0bt2aiIgI92chp/t63rx5rF69muHDhxMVFUVKSgpPPvkkrVu3JiwsjLFjxyIiXrfzLNmeNWsW4eHhhIWF8cADD2Q6r0ceeYTIyEguu+wy9+dEKXVuRGxJmLcq0KwmTID27eHmm2HbNti/H3r0sB0IJk60SVzZsv6OuACISJH4adWqlWT166+/uh/fdZdI584F+3PXXdleMpsrr7xSFi5cKCIikyZNknvvvVdERFJTU+Xo0aMiIrJ//3655JJLJCMjQ0REypcvLyIi27dvl5YtW4qIyPPPPy+jRo0SEZH169dLiRIlJCEhQUREDh48KCIiaWlp0rlzZ1m/fr2IiNSvX1/279/vjsX1fPXq1RIWFiZJSUly/PhxadGihfz000+yfft2KVGihKxdu1ZERAYPHizvv/9+tnM6dOiQO9bp06fLPffcIyIi999/v9zlcVEOHTok+/btk7p168q2bdsyxepp0qRJ8t///leOHj0qsbGx0rNnTxER6dKli/z222+ZrsPbb78tt912m3vfCRMmSLt27eTkyZOyf/9+qVq1qpw+fTrba2S9FiIikZGR8uOPP8r3338v5cqVc8eY2/UBZPny5SIiMmrUKJk8ebKkpKRI3bp1ZfPmzSIicv3118vUqVOzvW5CQoJ07tzZHffkyZOzxSkicvToUUlNTRURkf/9738ycOBA97k3bNhQjhw5IikpKVKvXj3ZsWOH7N69Wy6++GLZt2+fnDp1Stq3b5/pGrlkvXYiIgsWLJDevXuLiMiBAwfcyx955BF5+eWXRURkxIgR8tFHH7nX5bRdVt26dZOlS5fK5s2bJTw83L086/Fc9/tHH30kV1xxhaSnp8uePXukcuXK7u3q168vr776qoiI3H333RIeHi7Hjh2Tffv2SY0aNURE5KuvvpKbbrpJMjIyJD09Xfr06SNLlizJ9b7u3Lmz+3Mkkvn+vO6662TRokVet3M937Vrl/vap6amSteuXWXBggUiIgK497/vvvvkX//6l9fr5Pk9pZTK2++/i4DI9Om+bZ+YKFKpkkh0tEjDhiKhoSKzZ/s3Rl8Aq8XHHEdL1vzMsyrUswpURHj44YeJiIige/fu7Nq1K9f/vJcuXcp1110HQEREBBEREe51c+fOJSYmhujoaDZu3JhnldXy5cu5+uqrKV++PBUqVGDgwIHu0p2GDRsS5VTct2rVyuscnTt37qRXr16Eh4czefJkNm7cCMA333zDbbfd5t6uSpUq/Pjjj3Tq1ImGDRsCULVq1WzHi4uLY+nSpSxfvpw+ffqQlJREcnIyiYmJNG3aNNdzAejTpw9lypShevXq1KxZ0+cSDPEYhKdNmzbuGHO7PhdffLF7zszrrruO5cuXs3nzZho2bEiTJk0AGDFiBEuXLvUpBm+OHj3K4MGDCQsLY/z48e7rC9CtWzcqVapEaGgoLVq04M8//2TVqlV06dKFGjVqULp0aYYMGeLza3legw0bNhAXF0d4eDgzZ87M9LqefNlu7969bN26lY4dO9KkSRNKlizJhg0bco1l+fLlDB48mJCQEGrVquUuEXbp378/AOHh4bRt25aKFStSo0YNQkNDOXLkCF9//TVff/010dHRxMTE8Ntvv7FlyxbAt/sabFV527ZtCQ8P57vvvsvxGrgkJCS4r33JkiUZPny4+70vXbo0ffv2zfM1lVL54xoMN+uwHTmpXx9mzLA9Q1NSYMkSyMfXZFAoNjMYBGrKxwEDBnDPPffw008/kZKSQkxMDGAnRt+/fz9r1qyhVKlSNGjQgJMnT+Z6LOOlL/H27duZMmUKCQkJVKlShZEjR+Z5HM8/0FmVKVPG/bhEiRJeq0HvuOMO7rnnHvr378/ixYvdk2qLSLYYvS3LqnXr1qxevZpGjRrRo0cPDhw4wPTp02nVqlWu++UUc5oPIxkeP36cxMREmjRpwvr16zNNKp7b9cl6LsaYXLcvWbIkGRkZAHm+Ly6PPfYYXbt2ZcGCBSQmJtKlSxf3upzONa9rnJO1a9e656UcOXIkCxcuJDIyknfeeYfFixd73ceX7ebMmcPhw4fdCfCxY8eYPXs2Tz31VKZrIiLuye5zu45w5txDQkIyXYeQkBD3NGoPPfQQN998c6b9EhMTfbqvT548ybhx41i9ejUXX3wxEydOPKfPUqlSpdzvi6/3pVIqb/Hxtq1afpqNDhoEX34J4eFQp47/YvMXLVnzswoVKtClSxdGjx6dqWPB0aNHqVmzJqVKleL777/nzz//zPU4nTp1YubMmYAt2XC18Tp27Bjly5enUqVK7N27l//7v/9z71OxYkWOHz/u9VgLFy4kOTmZEydOsGDBAuLi4nw+p6NHj3LRRRcB8O6777qX9+zZk//+97/u54cPH6Zdu3YsWbLE3cD80KFD2Y5XunRpLr74YubOnctll11GXFwcU6ZM8RpTTueUH0lJSYwbN44BAwZQpUqVbOtzuz47duzghx9+AGxbpY4dO9KsWTMSExPZunUrAO+//z6dO3cGbFsr1zRfrjZ8eZ2H5/V955138jyftm3bsnjxYg4ePEhqaiofffSRT9dhyZIlTJs2jZtuugmwCWzt2rVJTU1132veYs1pO0+zZs3iyy+/JDExkcTERNasWeMuYfa8Jp988gmpqakAdOzYkfnz55ORkcHevXtzTBZz0qtXL9566y2SnH76u3btYt++fbnu43lursSsevXqJCUlZWpXl9P71bZtW5YsWcKBAwdIT09n1qxZ7vdeKeUf8fHQqpXtDZofvXoVzkQNNFk7L4YNG8b69esZOnSoe9nw4cNZvXo1sbGxzJw5k2bNmuV6jFtvvZWkpCQiIiJ47rnnaOO0rIyMjCQ6OpqWLVsyevRodxUdwNixY7niiiuyVSfFxMQwcuRI2rRpQ9u2bRkzZgzROfV/9mLixIkMHjyYuLg4qlev7l7+6KOPcvjwYcLCwoiMjOT777+nRo0aTJs2jYEDBxIZGZljFV1cXBwXXngh5cqVIy4ujp07d3pN1rp27cqvv/6aqYOBr7p27UpYWBht2rShXr16OU4mntv1ad68Oe+++y4REREcOnSIW2+9ldDQUN5++20GDx5MeHg4ISEh3HLLLQBMmDCBu+66i7i4OEp4fLP069ePBQsWeO1gcP/99/PQQw/RoUMHn3qn1q5dm4kTJ9KuXTu6d+/uLr31xtU5o0mTJjzzzDPMnz/fXbL2r3/9i7Zt29KjR49M9+PQoUOZPHky0dHR/PHHHzlu55KYmMiOHTu47LLL3MsaNmzIBRdcwKpVq7jppptYsmQJbdq0YdWqVe5SzWuuuYa6desSFhbGzTffTNu2balUqVKe5+/Ss2dPrr32Wtq1a0d4eDiDBg3KM7EfOXIkt9xyC1FRUZQpU4abbrqJ8PBwBgwYQGvXhINZtvMslatduzaTJk2ia9euREZGEhMTw1VXXeVzzEqp/Dl92lZn+tK5oCgxeVU9FBaxsbGSddyxTZs2uf8QKXWuEhMT6du3b55tr9TZS0pKokKFChw8eJA2bdqwYsUKatWqFeiw/Eq/p5Ty3erVduL2uXNh8OBAR3NujDFrRCTWl22LTZs1pVTw69u3L0eOHOH06dM89thjRT5RUyqYffmlHdYimGr2XZ0LilvJmiZrSvmoQYMGWqrmZ/ltp6aU8o9du+Caa6BaNdi+Pf/tw/wlPh5q1oR69QIdyfmlbdaUUkoplcnDD0NyMvz1F3z7baCjOcM1GG6wTbTub0U+WSsqbfKUUkWPfj+pYJSQAO+9B/fcA1WrwptvBjoi6+hR+O0338dXK0qKdDVoaGgoBw8epFq1amc9DpVSSvmDiHDw4EFCQ0MDHYpSbiJw991w4YV2Oqa0NHj9dTh40FaJBtKaNTa+4tZeDYp4sla3bl127tzJ/v37Ax2KUkplExoaSt26dQMdhlJuc+fCypW2NK1iRbjxRnj5ZZg5E+68M7CxuToXxPrUf7JoKdJDdyillFLKNykp0KyZLUFLSDjTqaB1azu+2bp1gW0rdvXVsHEj/P574GIoSPkZuqPIt1lTSimlVN5eeAF27ICpUzP3/hw9Gn7+2VZDBpKrc0FxpMmaUkopVczt3g2TJsHAgdnHVRs2DEJD4a23AhMb2KFEdu/WZE0ppZRSxdQjj0BqKjz3XPZ1lSvbidA//NBWlQZCcR0M10WTNaWUUqoYW7MG3nnH9gK95BLv29x4ox06Y/788xqaW3w8lCoFUVGBef1A82uyZozpbYzZbIzZaox50Mv6esaY740xa40xPxtjrvRY95Cz32ZjTC9/xqmUUkoVR66hOmrWtKVrOenUCRo1CtyYa/HxEBlpq2OLI78la8aYEsArwBVAC2CYMaZFls0eBeaKSDQwFHjV2beF87wl0Bt41TmeUkoppQrIvHmwfDk89RRccEHO24WE2I4GixfDH3+ct/AAyMiwvVOLaxUo+LdkrQ2wVUS2ichpYDZwVZZtBHDdHpWA3c7jq4DZInJKRLYDW53jKaWUUqoAnDwJ990HERE2EcvLiBE2aXv7bf/H5um33+D4cU3W/OUi4C+P5zudZZ4mAtcZY3YCXwB35GNfpZRSSp2lqVPhzz/hxRd9m6i9bl3o1cu2b0tP93t4bsW9cwH4N1nzNnRe1hF4hwHviEhd4ErgfWNMiI/7YowZa4xZbYxZrbMUKKWUUr7ZsweeeQYGDICuXX3f78Yb7TAaX3/tv9iyio+3syk0bXr+XjPY+DNZ2wlc7PG8LmeqOV1uBOYCiMgPQChQ3cd9EZFpIhIrIrE1atQowNCVUkqpouvRR+HUKZg8OX/79esHNWqc344G8fF2FoWQYjx+hT9PPQFobIxpaIwpje0wsCjLNjuAbgDGmObYZG2/s91QY0wZY0xDoDEQ78dYlVJKqWLhp59su7O77oJLL83fvqVLw/XXw6JFcD4qtE6ehPXri3cVKPgxWRORNOB24CtgE7bX50ZjzJPGmP7OZv8EbjLGrAdmASPF2ogtcfsV+BK4TUTOYw25UkopVfSkpNihOqpXt6VrZ2P0aDuA7vvvF2xs3qxbB2lp0Lat/18rmOlE7koppVQRd+AAvPoq/Oc/9vGMGbb92dm67DLbQ3PDBv9O7v7yy7YEcNcuqFPHf68TCDqRu1JKKaXYtg1uvx3q1YMJE2wJ1eLF55aogd3/11/P9NT0l1Wr4KKLil6ill+arCmllFJFzOrVMGQING4M06bZxxs2wGefZZ+o/WwMGQLlyuW/o8HWrbYK1Vfx8dpeDTRZU0oppYoEEfjiCzsUR+vW8OWXcO+9sH277VDQsmXBvdYFF8DgwTB7Npw4kff2v/xie5I2bmyTxb/+ynufQ4dscqfJmiZrSimlVKEjYntjLl9uS7fuvx/Cw6FPH9iyBaZMsQnRv/9tqxH94cYbbbu1efNy3iYxEW64wc7ruWyZrZL95ReIjoavvsr9+AkJ9rcma1Ay0AEopZRSxcnnn8PEibZhfvXqUK1azr+rVYMjR2Dz5uw/hw+fOWbp0jYhevddGDrUPve3jh1tSdmbb9qpqDzt32/nG33tNTs7wn33wQMPQNWqNmEbNAiuuAIeewwef9z7DArx8fYaxfrUBL9o02RNKaWUOg+OH4d//hOmT7ej8TdoAPv2waZNtodmUlLex6hTx+47ZIj97fqpX9+3KaMKkjF2GI+HHoLff4cmTew5vvCCLdlLSbHrH3/cTlXl0rSp7Tgwbhw8+SSsXAkzZ0LNmpmPHx8PzZvnPsF8caHJmlJKKeVnS5fCyJG2WvCBB+CJJ6BMmczbnDoFBw/anwMHzvyuWBGaNbPJUMWKgYg+ZyNG2PHaXn/dJp9PPWVL1a65xj5u1sz7fuXK2XZ0cXG2pC06GubMsaV1YKt54+PhyivP26kENU3WlFJKKT85edJW9T3/PDRsaJM2V0KSVZkytuSsMA1TUbu2TaimTrXPu3aFZ5/1rZ2ZMbbdW6tWtlq0Sxe77z//aSeY37dP26u5aLKmlFJK+cHatXZqpo0b4ZZb7DycFSoEOqqC98gjkJEBd94JPXrkf5DcqChYs8ZWmd53H6xYYduzgSZrLjqDgVJKKVWA0tJsCdETT5yZ9NyVfKicicBLL9mELT3ddpI4duz8dJYIBJ3BQCmllAqAzZttNedjj9lxyDZs0ETNV8bYeUuXLLHDjXTuXHQTtfzSalCllFLqLInYcc1WrLA/H34IZcvawWKHDAl0dIVT+/bwxx+2hFJZmqwppZQqMtLT7ZAR/mobdvKkncppxQo75MTKlbbHJkCVKjBggB22ojB1EghGpUtrqZonTdaUUkoVGUOHwnff2cnKw8ML5pgrVsCCBfb3mjVn5rZs0gT69oUOHexP06YQoo2LlB9osqaUUqpIWLjQTn1UujT07GmnYrrkknM75syZdrqkUqXsfJvjx9vErF0723lAqfNBkzWllFKF3vHjcMcdtjTtgw/g8suhe3ebsJ3t3Jjvv28Hsu3cGRYtKprDbqjCQQtslVJKFXqPPw67dsEbb0BEBHz5pZ0BoEePM23K8uPdd+3o/F27wmefaaKmAkuTNaWUUoXaTz/Byy/bgWfbtbPLYmPh009h+3bo3duO1+Wrt9+GUaNsydynn9qpkZQKJE3WlFJKFVrp6TB2rJ0E/JlnMq/r3Nm2YVu/Hvr1s71E8/Lmm3YKpB494JNP7DAcSgWaJmtKKaUKrVdesT00X3oJKlfOvr5PH3jvPVi2zA5S6+rJ6c306TBmDPTqpYmaCi6arCmllCqUdu6081L27m0TsZwMGwavvQaff257dqanZ9/mjTdsCd0VV9hhOkJD/Re3UvmlvUGVUkoVSnfeaROvV1/Ne/Lwm2+Go0fhgQegUiWbvLn2ee01GDfOlsLNnw9lyvg/dqXyQ5M1pZRShc6iRbYE7NlnoWFD3/a5/344fNjuU7my/f3KK3D77bZN20cfaaKmgpMma0oppQqVpCSbYIWFwT335G/fZ56BI0fg3/+2k6x//jlcdRXMnavTG6ngpcmaUkqpQmXCBPjrL5gzx84skB/G2NK0Y8fspOsDBtjjaKKmgpkma0oppQqNtWvhxRdtGzTXmGr5FRIC77wD119vZzrQRE0FO03WlFJKFQquMdVq1IBJk87tWKVK2V6kShUGmqwppZQqFF59FVavhlmzoEqVQEej1Pmj46wppZQKert22THVevWCIUMCHY1S55cma0oppYLe3Xfb2Qd8GVNNqaJGkzWllFJB7eef7RyfDz4IjRoFOhqlzj9N1pRSSgW155+H8uXtjAVKFUearCmllApaO3fa8dDGjNFOBar40mRNKaVU0PrPfyAjA+66K9CRKBU4mqwppZQKSsePwxtvwKBBvs//qVRRpMmaUkqpoPTmm3D0KNx7b6AjUSqwNFlTSikVdNLSYOpU6NQJWrcOdDRKBZbOYKCUUirozJsHO3bYNmtKFXdasqaUUiqoiMCUKdCkCfTtG+holAo8LVlTSikVVJYuhTVrbOeCEC1SUEpL1pRSSgWXKVOgRg24/vpAR6JUcNBkTSmlVNDYtAk++wxuuw3Klg10NEoFB03WlFJKBY2pUyE0FMaNC3QkSgUPTdaUUkoFhb174b33YORIWw2qlLI0WVNKKRUUXnkFTp+G8eMDHYlSwUWTNaWUUnnKyICxY+Gjj/xz/ORkePVV6N/fDtmhlDpDkzWllFJ5WrkSpk+HIUNsVWVBe/ddOHhQp5ZSyhsdZ00ppVSe5s61Df/btbNtykRgxIiCOXZ6OrzwArRtCx06FMwxlSpKtGRNKaVUrtLT7fRPV14Jn38O3bvDqFHw9tsFc/xFi2DrVluqZkzBHFOpokSTNaWUUrlasQL27IF//MOOffbJJ9CzJ9x4I7z55rkff8oUaNgQrr763I+lVFGkyZpSSqlczZ1rk7Q+fezzsmVh4ULo1QvGjIFp087+2D/8YNvDjR8PJUoUTLxKFTWarCmllMqRqwq0Tx+oUOHM8tBQWLDAVo3efLOdx/NsPP88VKliq1WVUt5pBwOllFI5WrbMDlb7j39kXxcaCh9/DIMGwS232OE9br3Vt+Nu22ZL5z7+GB56KHMiqJTKTJM1pZRSOZo7F8qVsyVo3pQpY0ve/vEPO0VURoad1zOrjAxISLCdCRYtgg0b7PKYGLjrLv/Fr1RR4NdkzRjTG3gJKAHMEJFns6yfCnR1npYDaopIZWddOvCLs26HiPT3Z6xKKaUyS0uD+fOhb18oXz7n7cqUsYPl/uMfcPvtNjG74w5ISYFvv7XJ2aefwt9/23ZpnTrZOUD79YNLLjl/56NUYeW3ZM0YUwJ4BegB7AQSjDGLRORX1zYiMt5j+zuAaI9DpIhIlL/iU0oplbulS2HfPu9VoFmVLm1L4YYOhTvvtElefLxN2CpWhCuugKuusr+rVPF/7EoVJf4sWWsDbBWRbQDGmNnAVcCvOWw/DJjgx3iUUkrlw9y5tkTtiit82750aZgzx05LtXy5Hdqjf3/o3NmuU0qdHX8maxcBf3k83wm09bahMaY+0BD4zmNxqDFmNZAGPCsiC/0VqFJKqcxcVaD9+tk2a74qVargBstVSln+HLrD2zjUksO2Q4F5IpLusayeiMQC1wIvGmOytWwwxow1xqw2xqzev3//uUeslFIKgMWL4cAB36pAlSoqZs6EBg0gJMT+njkz0BFZ/kzWdgIXezyvC+zOYduhwCzPBSKy2/m9DVhM5vZsrm2miUisiMTWqFGjIGJWSimFrc6sUAF69w50JEqdHzNn2ir8P/+0c9/++ad9HgwJmz+TtQSgsTGmoTGmNDYhW5R1I2NMU6AK8IPHsirGmDLO4+pAB3Ju66aUUoXaTz/BkCFw/HigI7FSU+34Z/3729kKlCoOHnkEkpMzL0tOtssDzW9t1kQkzRhzO/AVduiOt0RkozHmSWC1iLgSt2HAbBHxrCJtDrxhjMnAJpTPevYiVUqpomTRItuYv3p1eOWVQEcD330Hhw5pFagqXnbsyN/y88mv46yJyBfAF1mWPZ7l+UQv+60Ewv0Zm1JKBYvERPv71VdtCVunTgENh7lz7XAbvXoFNg6lzqd69WzVp7flgaZzgyqlVIBt3w7R0dCokR3uImtVzPl0+rSd8/Oqq+x0UkoVF08/nb3nc7lydnmgabKmlFIBlpgILVvCjBmwdSs8/nieu/jNt9/C4cNaBaqKn+HDYdo0qF8fjLG/p02zywNNkzWllAqg1FTYudMOE9C1q50QfepUWLUqMPHMnQuVKkHPnoF5faUCOXzG8OH2n6eMDPs7GBI10GRNKaUC6q+/7B+Ghg3t83//Gy66CEaPhlOnzm8srirQAQPsfJ9KnW/BPHxGIGmyppRSAbR9u/3tStYuuMBWvfz6Kzz11PmN5X//g6NHtQpUBU4wD58RSJqsKaVUALmStQYNzizr3RtGjIBJk2Dt2vMXy9y5ULkydO9+/l5TnV/BOkK/S36Gzwj2cylImqwppVQAJSZCiRJw8cWZl7/wAtSoYatDU1P9H8epU7BwIVx9tU66XlQVhirGnIbJyLo8v+dS2BM7TdaUUiqAtm+HunWhZJZRL6tWhddeg3Xr4Lnn/B/H11/DsWNaBVqUFYYqRl+Hz8jPuRSGJDUvmqwppVQAJSaeaa+W1YABNnl68knbhs2f5s6FKlWgWzf/vo4KnGAeod/F1+Ez8nMuhSFJzYsma0opFUDbt+ecrAH85z92NoHRoyE93T8xnDwJn3wCAwdCqVL+eQ0VeL5WMQaaL8Nn5OdcCkOSmhdN1pRSKkBOnoQ9ezJ3LsiqZk2bsK1aBS+95J84vvrKTiKvVaBFWzCP0J9f+TmXwpKk5kaTNaWUChDXPIS5lawBDB0K/fvbapstWwo+jrlzoVo1OyivCh4F3Sg+mEfoz6/8nEtRSFI1WVNKqQDxNmyHN8bYzgZlysCYMbZ6qKCkpMCiRVoFGmz81Si+oEfoLwyzDRSFJLVk3psopZTytGcP1Kplv/jPRWKi/Z1XyRpAnTp2OI8bb7TziMbFQfv20KEDXHqp77GI2NdduRJWrIClSyEpCQYPPtuzUP6QW6P4YEkyXAmlK05XQgnBE6PL8OHBF1N+aMmaUkrlw44d9j/z+fPP/Vjbt9vSrNq1fdt+1Cj473+hUSP46CP7vEkTuPBC23P0uedsAnby5Jl9Tp+27d1eeAEGDbJTWTVqBNddBx98YJ+/9JIOhBtsCsPgsEWhl2VhoSVrSimVDytX2kFq4+Nt8nMuEhNt4leihG/bGwO33WZ/MjJg06YzJWQrV9oenWAHtW3Vyo7dlpBwJnlr2BAuv9yWxnXoYEvofH1tdX7Vq3emTWPW5Z4CWbqV316WM2faRG7HDnseTz9duEu7zidN1pRSKh8SEuzvTZvO/Vh5DduRm5AQm2y1bAk33WSX7dsHP/xwJnk7fRpuvdUmZu3b+16CpwLv6aczJ2GQ/8Fh/Z0I+ZpQQuGqMg1GRhhO9j0AACAASURBVEQCHUOBiI2NldWrVwc6DKVUERcXB8uXwyWXwNat53asGjXs9E7TphVMbKpo8aUkKiTEtkPMypiC7YiSU3zeEkpvjfcbNPCe2NWvf6btZnFjjFkjIrG+bKtt1pRSykdpafDTT7bqcPv2zG3D8ispCQ4cOPuSNVX0FfTgsAUtP70si8LAtIGkyZpSSvlo0yZbitC7t/0D+vvvZ38sV2lCXsN2KJWbQI8h5uvwGUVhYNpA0mRNKaV85GqvdsMN9ve5tFvLz7AdSuWksIwhFuiksrDTZE0ppXwUHw8XXAB9+9o/jOeSrLkGxNVkTZ2rgh7oFnT2hGCjvUGVUspHCQkQG2tLBBo2hN9+O/tjJSZC2bJ27k+lgom/em4W9oFpA0lL1pRSygcnT8LPP0ObNvZ58+bnXrLWoMG5z4KgVEHTwW6DjyZrSinlg/XrbW/Q1q3t8+bNYfNmSE8/u+O5kjWlgo323Aw+mqwppZQP4uPtb89k7dSpsx8jKjFR26sVV4Gc/NwX2nMz+OSZrBljbjfGVDkfwSilVLBKSLBzcNata583b25/n01V6JEj9keTteLH1R7szz/tYLau9mDBlLBpz83g40vJWi0gwRgz1xjT2xhtYaGUKn4SEmx7Ndc34LkkazrGWvFVGNqDac/N4JNnsiYijwKNgTeBkcAWY8wzxphL/BybUkoFhaNHbc9PVxUoQOXKUKvW2SVrOmxH8VVY2oP5YzgQdfZ8arMmdgLRv52fNKAKMM8Y85wfY1NKqaCwZo397Zmswdn3CNWSteJL24Ops+FLm7U7jTFrgOeAFUC4iNwKtAKu8XN8SikVcK6ZC7Ima82a2WTN20Taudm+HSpWhKpVCyY+VXhoezB1NnwZFLc6MFBE/vRcKCIZxpi+/glLKaWCR0ICNGoE1aplXt68ua0i/ftvqF3b9+Nt326rQLUFcPHjqk585BFb9Vmvnk3UtJpR5caXatAvgEOuJ8aYisaYtgAicg5DQiqlVOEQH5+9VA3OvpNBYqJWgfpDsA+J4aLtwVR++ZKsvQYkeTw/4SxTSqkib+9e+OuvgkvWRM6UrKmC468hMQpLAqiKNl+SNeN0MABs9Sc6p6hSqphwtVdzTTPlqU4d2/YsP8nawYNw4oSWrBU0fwyJURjGRFPFgy/J2jank0Ep5+cuYJu/A1NKqWCQkGBLVWJisq8zJv89QnXYDv/wx5AY+UkAtQRO+ZMvydotQHtgF7ATaAuM9WdQSikVLOLjoUULKF/e+/rmze0YbL5yJWtaslaw/DEkhq8JoJbAKX/zZVDcfSIyVERqisiFInKtiOw7H8EppVQgiZyZuSAnzZvD7t22V6gvXGOsaclawfLHkBi+JoCFYVYCVbj5Ms5aqDHmNmPMq8aYt1w/5yM4pZQKpMRE28bMW+cCF1cnA19L17Zvt+OrXXDBOYenPPhjiiRfE8DCMiuBKrx8qQZ9Hzs/aC9gCVAXOO7PoJRSKhjEx9vfviRrvrZb02E7/Cc/Q2L40sbM1wRQZyVQ/uZLsnapiDwGnBCRd4E+QLh/w1JKqcBLSIDSpSE8l2+8hg3tNr4mazpsR+Dlp42ZLwmgzkqg/M2XZC3V+X3EGBMGVAIa+C0ipZQKEgkJEB1tk7GclCwJTZr4lqy5EgMtWQusgm5j5o8qWKU8+TJe2jRjTBXgUWARUAF4zK9RKaVUgKWn2wncR43Ke9vmzWHt2ry3+/tvOHlSS9YCzR9tzIYP1+RM+U+uJWvGmBDgmIgcFpGlItLI6RX6xnmKTymlAmLTJjt4bW7t1VyaNYNt22wilhsdYy04aBszVdjkmqw5sxXcfp5iUUqpoJHbzAVZNW9u2zRt2ZL7dq5hO7QaNLC0jZkqbHxps/Y/Y8y9xpiLjTFVXT9+j0wppQIoIcEOr9GkSd7b+tojVAfEDQ7axkwVNr60WRvt/L7NY5kAjQo+HKWUCg7x8dCqlR3aIS9Nm9o/+nkla4mJULNm9lIddf5pGzNVmOSZrImItq5QShUrp07Bzz/D+PG+bV+2rC0t86VkTdurKaXyK89kzRhzg7flIvJewYejlFKBt349pKb61l7NxZcJ3bdvz98xlVIKfGuz1trjJw6YCPT3Y0xKKRVQrs4FvvQEdWneHDZvtkN+eJOeboeGKOrt1XyZGUAplT++VIPe4fncGFMJOwWVUkoVSfHxtm3ZxRf7vk/z5rb69M8/oZGXFr27dkFaWtGuBnXNDOAacNY1MwBo+zClzoUvJWtZJQONCzoQpZQKFgkJtrrSGN/3yatHaHEYtqOgZwZQSlm+tFn7FNv7E2xy1wKY68+glFIqUI4fh99+g6FD87efZ7LWp0/29cVhQFx/zAyglPJt6I4pHo/TgD9FZKef4lFKqYBas8bO4Zmf9mr8f3t3Hi9VXf9x/PXhomLhwiJuILiA4kJKhAZlKgm4V2ZCaIaWLZBaWi4tmkpphktKiRaaippmGlmJQLjhwoVcL7iAAiL+WBQUXNju9/fHZyaGy9zLzL3nzDkz834+Hvcxd849c873dgw+fL+f7+cDtGsHO+7Y9MyaWWVXyd9tN1/6zHdcRJqvkGXQBcAzIYRHQwjTgHfMrFusoxIRScj06f5abLAGTe8IfeMN2GUX2Gqr5o8t7dQZQCQehQRr9wL1Oe/XZ45tlpkNNrNXzGyOmV2Q5+fXmNlzma9XzWxFzs9OM7PXMl+nFXI/EZGWqq31pcqOHYv/bDZYC2HTn1VDjTV1BhCJRyHLoK1DCGuyb0IIa8xsy819yMxqgDHAkcBCoNbMJoQQZuVc64c55/8AOCjzfXvgYqAPni83M/PZ5YX9WiIizVNbCwcf3LzP7rMPrFgBixfDTjtt/LN58+DQQ1s8vNRTZwCR6BUys7bUzP5XV83MTgCWFfC5vsCcEMLrmWDvbuCEJs4fCtyV+X4QMCmE8G4mQJsEDC7gniIizbZkiedcNWcJFBrfEbp2LSxcWPkzayISj0KCte8CF5nZAjNbAJwPfKeAz+0KvJnzfmHm2CbMrCuwO/CfYj5rZmea2Qwzm7F06dIChiQi0rhsMdzmdhloLFh7802or6/ssh0iEp9CiuLOBQ4xs7aAhRBWFnjtfBWK8mRyADAE+GsIIVv7u6DPhhBuAm4C6NOnT2PXFhEpSG2tV97v3bt5n991V9hmm02DtWoo2yEi8dnszJqZ/crMtg8hrAohrDSzdmZ2eQHXXgjk1v/uDCxq5NwhbFgCLfazIiKRqK312bG2bZv3eTPPW2sYrGUL4ipYE5HmKGQZ9KgQwv92aWZyyI4u4HO1QHcz2z2zIWEIMKHhSWa2N9AOeCrn8ERgYCYwbAcMzBwTEYlFCF62o7n5aln5yne88QbU1EDnzi27tohUp0KCtRoz+19lIDPbGthspaAQwjpgJB5kzQbuCSHUmdmluRsW8I0Fd4ewYbN7COFd4DI84KsFLs0cExGJxfz5sGxZ8/PVsnr2hEWL4L33Nhx74w3vM9q6kP33sgk1h5dqV8gfHXcAU8zslsz74cCfC7l4COFfwL8aHPtFg/eXNPLZccC4Qu4jItJS2c0FUcysgbesypYAmTdPmwuaS83hRQqYWQsh/Aa4HOiJ9wV9COga87hEREpq3DjYbjvo1atl18kN1rKqoSBusQqdLVNzeJHClkEB/g/vYnAiMABf1hQRKcjpp8N11yU9isb9+9/w0EPwi1/Alpst+d20Pfbwa2Tz1j7+GN5+W8Faruxs2fz5niuYnS3LF7CpObxIE8GamfUws1+Y2WzgBrzumYUQDg8h3FCyEYpIWVuzBm6/Hc4/P3+T76StXQvnngt77QUjR7b8eq1bQ/fuG4K17O+sZdANipkta6wJvJrDSzVpambtZXwW7bgQwudCCNfjfUFFRAr26quwbh2sXu0BW9qMHeuB1ejRLZ9Vy8rdEaoaa5sqZrZMzeFFmg7WTsSXP6ea2c1mNoD8xWpFRBpVV+evX/oS/OUvMG1asuPJ9e67cPHFMGAAHHdcdNft2RPmzvUANVtjTTNrGxQzW6bm8CJNBGshhPtDCCcD+wCPAD8EdjSzP5jZwBKNT0TK3EsveY2xP/0JdtkFzjnHWy+lwaWXeuP1q6/2QCAq++zjv+Nrr/nM2pZb+u8urtjZsmHDPOitr/dXBWpSbQrZDfpBCGF8COFYvJPAc8AFsY9MRCpCXZ3ng7VvD1dcATNmwB13JD0q3605Zgx8+9st3wHaUG6P0HnzfDaoVaHbuaqAZstEimM5tWjLWp8+fcKMGTOSHoaINLD33rD//nDffT4zcsgh8NZbnsv2yU8mN65jj4XHH/fZr06dor32hx96y6pLLoEHH4Ttt4eHH472HiJS3sxsZgihTyHn6t96IhKbjz+GOXNgv/38fatWcO21XuH/N79JblwTJ8I//wk/+1n0gRr4kl7Xrhtm1rS5QERaQsGaiMTmlVd8Ni0brAH06wdDhniwlkStrHXr4Ec/8npoZ50V33169vQl36VLS7+5QO2ZRCqLgjURiU12J2husAaeuwZw4YWlHQ94btSsWfDb38JWm+1y3Hw9e/qsIpR2Zq2YgrMiUh4UrIlIbOrqvEhsjx4bH+/aFc47D+68E556qnTjWb7cuxQcdpiXEolTdpMBlDZYU3smkcqjYE1EYlNX59X88xWbPf982Hln+OEPS1fK47LLvLbaNddEW6ojn9xgrZTLoGrPJFJ5FKyJSGzq6jZdAs1q2xZ+/Wt45hm46674x/Lqq3D99XDGGXDggfHfLxusbb11PJsYGhNXeyblwYkkR8GaiMTio4+8in9jwRrAqafCpz/ts2wffBDveH78Yw+cLr883vtktW/vQVq3bvHP4uWKoz2T8uBEkqVgTURi8fLL/hd7U8FatpTHW295wn9cJk+GCRM8b2vHHeO7T0MDB8IXvlC6+0E8BWeVByeSLBXFFZFY3HGHz5zV1cG++zZ97sknwz/+4UuVnTtHO4516+Cgg3zmbtYsaNMm2utXg1atPPBuyCw9rcNEyo2K4opI4urqYIstfIPB5lx5pf+lf9FF0Y/jT3/y/qRXXaVArbniyoMTkcIoWBORWNTVecmOLbbY/LndusG558Ltt8P06dGNIQQPBPv3h698JbrrVps48uBEpHAK1kQkFk3tBM3nggtgp528u0BU5s6FN96AoUNLm+RfadR4XSRZCtZEJHIffuhBUjHB2jbbeKHcadO8n2YUJk/21yOPjOZ61WzYMH8u9fX+qkBNpHQUrIlI5GbP3vxO0HyOOcZfJ06MZhyTJkGXLoXlzVUj1U4TKQ8K1kQkco31BN2cvff2JbaHHmr5GNavh//8x2fVtAS6KdVOEykfCtZEJHJ1dd5iaq+9ivucGQwaBFOmwNq1LRvDzJmwYgV88Ystu06lUu00kfKhYE1EIldX57NkrVsX/9nBg2HlSnj66ZaNIZuvNmBAy65TqdRDVKR8KFgTkcgVuxM01xFHQE1Ny5dCJ02CT32qtH05y4lqp4mUDwVrIhKpVat8t2Bzg7XttoN+/Vq2yeCDD+DJJ7ULtCmqnSZSPhSsiUikZs/21+YGa+B5azNnwpIlzfv844/DmjXKV2uKaqeJlA8FayISqebuBM01aJC/TprUvM9PnuwbHD7/+eaPoRqodppIeVCwJiKRqquDrbaCPfds/jV694aOHZuftzZpkreYarjMJyJSjhSsiUik6upgn318k0BztWoFAwfCww/7rE8xFi+GF15QvpqIVA4FayISqZbsBM01eLDnrD3/fHGfmzLFX5WvJiKVQsGaiERm5Uqv0xVFsDZwoL8WuxQ6eTK0a+dLqSIilUDBmohEZtYsf40iWNtxRzjwwOJKeITg+WrZWm2VRr08RaqTgjURiUwUO0FzDR4M06bB++8Xdv6rr8LChZWZr6ZeniLVS8GaiESmrg7atIHdd4/meoMGwbp1MHVqYednS32UU75aobNl6uUpUr0UrIlIZOrqoGfP6JYg+/WDtm0Lz1ubPNkDxZaUDSmlYmbL1MtTpHopWBOpErfe6iUt4lRXB/vvH931ttzS888mTvRgpinZGbhymlUrZrZMvTxFqpeCNZEqMG4cDB/ugUxcMzHvvef5YlHlq2UNGgRvvAFz5jR9Xm2t57aVU75aMbNl6uUpUr0UrIlUuGefhREj4LOfhdWr4YQTvNF51KLcCZpr8GB/3dxS6OTJ3uPyiCOivX+cipktUy9PkeqlYE2kgq1YAV/9KnToAA88AHfd5UVmhw/f/LJisaLeCZq1xx6w116bL+ExaZLXVuvQIdr7x6nY2TL18hSpTgrWRCpUfT2cdpovqd17L3TqBEcfDVde6e+jXj576SUPNLp2jfa64LNrU6f6zGA+q1bBU0+VV74aaLZMRAqjYE2kQl11FUyYAKNH+xJo1nnnwSmnwM9/DvffH9396upg3329BEXUBg3yxPsnnsj/80cf9Q0G5ZSvlqXZMhHZHAVrIhVo6lS46CL42tfgBz/Y+GdmcPPN0LcvnHoqvPhiNPeMqidoPocd5jtDG8tbmzzZ67v17x/P/UVEkqRgTaTCLFoEQ4ZAjx7wxz96cNZQmzY+q7bttnD88bBsWcvuuXw5vP12fMFa27bwuc81nrc2aRJ8/vP+e6WFWkOJSFQUrIlUkLVrfTbtgw/gvvtgm20aP3eXXXzTwdtvw0kn+WebK67NBbkGDfJZwEWLNj7+9tt+/zTlq6k1lIhEScGaSAW54ALvpXnzzZ4/tjl9+/q5jzwC55zT/PuWIljLlvBoOLs2ebK/pilfTa2hRCRKCtZEKsRf/wpXXw0jR8LQoYV/7tRT4cc/ht//HsaObd696+p8qTLOavoHHAA775w/WOvYET71qU0/U+hSZNRLlmoNJSJRUrAmUgFefRVOPx0OPth3fxbr17+Go47yQO+xx4r/fHYnaL78uKiY+VLoww/D+vV+LATPVxswYNNdqIUuRcaxZKnWUCISJQVrImXugw/gxBN9t+S99/prsWpqvGDunnv6tebNK+7zce4EzTVokG9mmDHD38+e7Tlr+fLVCl2KjGPJUq2hRCRKCtZEylgI8N3verB0113QpUvzr7Xddl6Xbe1ab0m1alVhn3vnHVi8uDTB2pFH+gxbtoTHpEkbjjdU6FJkHEuWKnYrIlFSsCZSptat800Bd9wBv/xlNAn2PXrAPfd4N4LTTvNCrZtTis0FWR06wGc+syFvbfJkb0WVr2tCoUuRxS5ZFprfpmK3IhIVBWsiZei99+C44+B3v/OALcpdhgMHwm9/C3/7G1x22ebPL2WwBr4U+swzsGSJ72JtLEgtdCmymCVLleQQkSQoWBMpM6+/Dv36+azS2LFwzTXRt3g65xz45jfhkku8XltT6uq8uG7nztGOoTGDB/ts1ahRvlTbWH21Qpcii1myVEkOEUmChRDiu7jZYOA6oAb4YwjhijznfA24BAjA8yGEr2eOrweyjXAWhBCOb+peffr0CTOyWcciFerxx+ErX/HdkPfdB4cfHt+9Vq/2Nk8vvABPPpm/NAb4GD7+2Bupl8K6dV6q48MP/X+HZcugXbvS3LtVK59Ra8issCVjEZEsM5sZQuhTyLmxzayZWQ0wBjgK2BcYamb7NjinO3Ah0D+EsB+QW5bzoxDCgZmvJgM1kWpw661eoqJ9e18GjDNQA9hqK18KbdfOW1ItWZL/vFLtBM1q3dqXPteuhT59SheogUpyiEgy4lwG7QvMCSG8HkJYA9wNnNDgnG8DY0IIywFCCI38dSBSverr4fzzYfhwOPRQePpp6N69NPfeeWdvSbVkCXz1q7BmzcY/X7rUv0oZrIHnrUHpuxaoJIeIJCHOYG1X4M2c9wszx3L1AHqY2TQzezqzbJrVxsxmZI5/Kd8NzOzMzDkzli5dGu3oRVJg1Spf9vzNb+B734N//7u0M0ngs1fjxvkS7MiRGy8DlnpzQdbxx8MhhxTXqSEKKskhIkloHeO189Uyb5jt0RroDhwGdAYeN7P9QwgrgN1CCIvMbA/gP2b2Yghh7kYXC+Em4CbwnLWofwGRJL35pu/4fPFF3/U5cmS8HQKaMnSo565dcYXnro0Y4ceTCtY6dSpdjlxDw4YpOBOR0oozWFsI5Jbo7AwsynPO0yGEtcAbZvYKHrzVhhAWAYQQXjezR4CDgLmIlKH167122TPPwMqVmz9/3Tq49lpPov/nPzc0MU/S5Zf773D22dCzJxxxhAdr220Hu+yS9OhERCpXnMFaLdDdzHYH3gKGAF9vcM4DwFDgVjPriC+Lvm5m7YAPQwirM8f7A7+JcawikVq50nPLpk3znZRPP11YkJZrr71gyhTvuZkGNTVeT+yQQ+Ckk6C2dsPmgqZm/MaP99IWCxZ4Iv6oUZqZEhEpRmzBWghhnZmNBCbipTvGhRDqzOxSYEYIYULmZwPNbBawHvhxCOEdM+sHjDWzejyv7ooQwqy4xirSUvPne2CW/XrxRd8Y0KoVHHAAnHqq10br18/LThTiE5/wAClNtt3WW1L17et5Y4sW+caDxmSLyGZrk2WLyIICNhGRQsVaZ62UVGdNSm35cu/HecstGxqLb7ONzzz16wf9+8PBB3uAU2kmT/al2fXrfbn27LPzn9etmwdoDXXtWnyzeBGRSlJMnbU4l0FFKk59vS9N3nKL1yBbvRp69YLRoz2H64AD0jcbFocvfhGuvtqDtM98pvHz4miSLiJSbRSsSardfrv3wRwxIrmdkOAtnm69Ff78Zw802rWDb30LTj8dDjoo2bEl5ayz4Mtfhi5dGj9nt93yz6ypiKyISOEUrEmq/fSnXsKirg5uuKG0s1YffeQtncaNg6lTPSA78kiveXbCCdCmTenGklZNBWrgmwlyc9ZARWRFRIqlRu6SWosXe6C2335w441e62v16vjvu3Yt/OEPsMcevjFg/ny47DLPsZo4EU4+WYFaoVREVkSk5TSzJqlVW+uvv/+9J/Cfey68+y7cf78n8ketvh7uuQd+9jOYOxc+9znfzXjYYb6rU5pHRWRFRFpGwZqkVm2tB0m9e3tPzI4dPUfs8MO97dIOO0RznxDg4Yfhwgvh2Wd9k8CDD8LRR1dnLpqIiKSL5gsktaZP94Kwbdv6+298A/7+d5g1y2e9oij9MH06DBjgZSiWL/cNDc8+C8cco0BNRETSQcGapFIIPrPWsCzEMcfApEmwZInXMXvppeZd/+WXvZjrwQf7Na67zo+dckp1lN4od+PHew23Vq38dfz4pEckIhIfBWuSSvPmwTvv5K/h1b8/PPaYB3SHHurtnDYnBL/m+PE+Q7fffr5Z4JJLPD/trLNgq60i/iUkFtmuCPPn+3PNdkVQwCYilUrBmjSpvt4T+2++ubT3zW4u6Ns3/88POMCDtI4dvUDrv/618c/XrvUlzmuv9T6WnTvD7rv7zNn998MPfuC10y6+OJ7NChKfn/5041Ig4O9/+tNkxiMiEjdtMJAmXXKJV6rfbjsvnZHNH4vb9Omw5ZYelDWmWzd44gk46ijvU3nZZd4sfdo0D/Y++sjP69rVd3RmW0Dtvz+01n/5ZUtdEUSk2mhmTRp1770eAH3+895F4I47Snfv2lo48EAP2JrSqZMXrD30ULjoIrjqKg/SvvMdL8OxcOGG5c8RI/yahQRqyolKr8a6H6grgohUKs0vSF7PPgunnQaf/awn9Pfr5x0EvvOd+HdJrl8PM2fC8OGFnb/ttp5/VlcHPXp4hfyWyOZEZZfasjlRoHphaaCuCCJSbTSzJptYvNjbKXXo4M3Kt9oKRo70YOjRR+O//8svwwcfNN0gvKEttvBZs5YGaqCcqLRTVwQRqTYK1mQja9bAiSfCsmVe02ynnfz4kCHQvr3PrsVt+nR/LSZYK1Qhy5vKiUq/YcN8ebu+3l8VqIlIJVOwJv8Tgud1TZsGt9zinQOytt4avvUteOAB79cZp9pa36G5997RXrfQkg9J50QpX05ERHIpWJP/GTMG/vhHT9Q/+eRNf/6973mQc+ON8Y6jthb69Im+H2ehy5ujRm26nFqqnKhyqiGmoFJEpDQUrAkAU6bAOefAccf5DtB8unXzn990E3z8cTzjWL0ann8+niXQQpc3k8yJKpd8uXIKKkVEyp2CNWHuXC8cu88+Xp6jqRmtkSM9n+3ee+MZy/PPe0HbOIK1YpY3k8qJKpd8uTiCSs3UiYjkp2Ctyr3/vheUNfMNBdtu2/T5AwZ4UHf99fGMZ3OdC1oiyeXNQiWdL1eoqINKzdSJiDROwVoVq6/39kuvvOIFZPfcc/OfMfPZtdraDbs2o1Rb64Vuu3SJ/trlUPKhHAJKiD6oLJflXxGRJChYq2I//zn84x/eP3PAgMI/941v+G7NOMp41Nb6EmhchXfTXvKhHAJKiD6oLJflXxGRJChYq0IrV8K3vw2/+pW/jhhR3Oe32ca7G/zlL7BkSbTjmj07nny1clJoQBlHjleh14w6qCyX5V8RkSQoWKsyjz0GvXrBuHFw/vlerqM5s1gjRngB3Ztvjm5sM2d6vlIc+WqVJo4cr2KvGeUsZbks/4qIJEHBWpX4+GM47zw47DCoqfGg7YorvE1Tc+yzDxx5JPzhD7BuXTRjzG4uqPaZtULEkeOVZN5YuSz/iogkQcFaFfjvf73I7OjR3oj9ueegf/+WX3fkSHjrLd9FGoXaWl9669gxmutVsjhyvJLOG0t7PqGISFIUrFWwdevg8svh4INh+XL49799Jqxt22iuf8wxPgMSVRmP6dO1BFqoOHK8lDcmIpJOCtYq1Cuv+OzZz3/uBW9ffBEGD472HjU18P3vw6OP+vVbYulSz5HSEmhh4sjxUt6YiEg6KVirMPX1PtN10EEwZw7cfTfceSe0bx/P/c44A9q08Y0KLaF8teLEkeOlvDERkXRSsFZB3nwTBg6Es87ybM6vbgAADkZJREFUjQQvvpi/IXuUOnSAr38dbr8dVqxo/nVqaz1A6N07urGlTdSlNuLI8VLemIhI+ihYqwAhwG23wf77w9NPw9ix8M9/wi67lOb+I0b4rsFbbmn+NaZPh3339RpulUjtlEREpLkUrJW5pUvhxBO9SG2vXvDCCx4ExNUBIJ/evaFfP18Kra8v/vMhbOhcUKnUTklERJpLwVoZ+/vfYb/9fBbtqqvgkUdgjz2SGcvIkTB3LkycuPlzGy4H/u53HnRWcrCWdFkMEREpXwrWytB778Hw4fClL8Guu3rl//PO892ZSTnxRNhpp833C823HPiTn/jPKrlsh8piiIhIcylYS9BHH3lZhBtvhKlTYdEiD2CaMnWqL3fedhv87GfwzDOeq1asqJPdt9wSvvtd+Ne/fEyNybccuGaNv/bq1bIxpJnKYoiISHNZ2Fx0UCb69OkTZsyYkfQwinLNNfCjH218bJttoEcP2Hvvjb+6dIHLLoPrroPu3T1YO+SQ5t03O7uVGzR94hMtL9Pw/vvehmqXXTxgyzfT16pV4wFphfyn2Kjx4z1YXbDAZ9RGjdJuSxGRamVmM0MIfQo6V8FaMtatgz339FpW48d7EduGXwsWbBrAjBzpPT0/+cnm37tbN19+bKhrVy/X0BLjx8Mpp/iO1DPPLPzebdvCypUtu7eIiEi5ULBWBu6+G4YO9U0Cxx+f/5yPPoLXXvPAbc4c+OxnvX5aSzU2u2XWvN2cuULwMb70Erz6qtdhy5VvVg/82NixLbu3iIhIuSgmWFPOWgJC8N2bPXrAscc2ft7WW3se10knwYUXRhOoQbzJ7mbeQeG99zynrqGGVfKzwdzZZ296btR5dSIiIuVIwVoCHn0U/vtfOPdcD0RKLe5k9169vFDu2LH+ezaUWyV/yBBfAt17743PURFZERERp2AtAaNHww47wKmnJnP/UvSA/OUv/XccMaLppdXaWvj0pzfdjFBMEVnNwImISCVTsFZis2fDgw96ELP11smNI+4ekNtvD1de6e2vbrst/zlr1sBzz+Wvr1ZoEVnNwImISKVTsFZiV18NbdrA97+f9Eji941v+KaIn/wkf5P3F17wgC1f54JC8+rUxklERCqdgrUSWrwYbr8dvvlNXyKsdK1aeUeDZcvg4os3/Xltrb/mC9YKzatTG6fCaKlYRKR8KVgroTFjfCbphz9MeiSl07u3dza44QafSctVW+tBa9eum36u0Ly6pNs4lUMQpKViEZHypmCtRD780IO144/3kh1xSDJwaOrel18O7dp5nl5ufbfp031WzSz/NQvJq0uyjVO5BEFaKhYRKW8K1krk1lvh3Xe94XockgwcNnfv9u3h17+GJ56AO+/0Y6tW+WaLfEugxSjFztbGlEsQpKViEZHypg4GJbB+vffM7NABnnqq8ZmkloizhVQU966v916mb77pHRmeew6+8AXfGXvMMfGOLy5xdoKIUpL/bYiISH7qYJAyEyZ4u6jzzosnUINkZ08KuXd2s8HixXDppU1vLigXSefLFSrJpWIREWk5BWsl8Nvfwu67w5e/HN89kgwcCr13375wxhlw3XVw110+s9OpU/zji0u5BEFJLhWLiEjLKViL2VNPwZNP+g7QhlX6o5Rk4FDMvX/1K28vNXNmec+qQXkFQXEXQRYRkfgoWIvZ6NG+E3L48Hjvk2TgUMy9d9hhQxBX7sEaKAgSEZH4aYNBjObOhe7d4cIL07c0lqT1673J+9e+Bh07Jj0aERGR0itmg0HruAdTza65BrbYAkaOTHok6VJTUx3ttkRERKIQ6zKomQ02s1fMbI6ZXdDIOV8zs1lmVmdmd+YcP83MXst8nRbnOAtRbMHZd96BW27xZbGddy7FCONRDhX6RUREKllsM2tmVgOMAY4EFgK1ZjYhhDAr55zuwIVA/xDCcjPrlDneHrgY6AMEYGbms8vjGm9TskVfswVQs0VfofEcpRtv9PPPPbc0Y4xDc35vERERiVacM2t9gTkhhNdDCGuAu4ETGpzzbWBMNggLISzJHB8ETAohvJv52SRgcIxjbVKxleo//hiuvx6OOgr22y/+8cWlXCr0i4iIVLI4c9Z2Bd7Meb8QOLjBOT0AzGwaUANcEkJ4qJHP7trwBmZ2JnAmwG4xFhRrrOjr/Pn5Z5iWLPHir3G1lioVtSkSERFJXpzBWr5a/Q23nrYGugOHAZ2Bx81s/wI/SwjhJuAm8N2gLRlsU3bbLX+7ntatvRl5PiedBIcfHteISqOx3zttFfpFREQqWZzLoAuBLjnvOwOL8pzz9xDC2hDCG8ArePBWyGdLprGir7feCq+9lv/rnnviay1VKuVSoV9ERKSSxRms1QLdzWx3M9sSGAJMaHDOA8DhAGbWEV8WfR2YCAw0s3Zm1g4YmDmWiHKqVB+lav29RURE0iTWorhmdjRwLZ6PNi6EMMrMLgVmhBAmmJkBo/HNA+uBUSGEuzOfPR24KHOpUSGEW5q6VxqL4oqIiIjkU0xRXHUwSND48b6zcsECzwMbNUqzViIiItWgmGBNvUETkq1hNn8+hLChhlm+orMqTCsiIlK9FKwlpNAaZsUEdSIiIlJ5FKxFrNBZsEJrmKkwrYiISHVTsBahYmbBGqtV1vC4CtOKiIhUNwVrESpmFqzQGmaFBnUiIiJSmRSsRaiYWbBCa5ipMK2IiEh1U7AWoWJnwYYNg3nzoL7eX/OV7VBhWhERkeqmYC1Ccc2CFRLUiYiISGVSsBYhzYKJiIhI1FonPYBKM2yYgjMRERGJjmbWRERERFJMwZqIiIhIiilYExEREUkxBWsiIiIiKaZgTURERCTFFKyJiIiIpJiCNREREZEUU7AmIiIikmIK1kRERERSTMGaiIiISIpZCCHpMUTCzJYC85v58Y7AsgiHI9HS80k3PZ/00rNJNz2f9CrFs+kaQtihkBMrJlhrCTObEULok/Q4JD89n3TT80kvPZt00/NJr7Q9Gy2DioiIiKSYgjURERGRFFOw5m5KegDSJD2fdNPzSS89m3TT80mvVD0b5ayJiIiIpJhm1kRERERSrOqDNTMbbGavmNkcM7sg6fFUOzMbZ2ZLzOylnGPtzWySmb2WeW2X5BirlZl1MbOpZjbbzOrM7OzMcT2fFDCzNmY23cyezzyfX2aO725mz2Sez1/MbMukx1qtzKzGzJ41swcz7/VsUsLM5pnZi2b2nJnNyBxLzZ9tVR2smVkNMAY4CtgXGGpm+yY7qqp3KzC4wbELgCkhhO7AlMx7Kb11wLkhhJ7AIcCIzP9f9HzSYTVwRAjhU8CBwGAzOwS4Ergm83yWA2ckOMZqdzYwO+e9nk26HB5CODCnZEdq/myr6mAN6AvMCSG8HkJYA9wNnJDwmKpaCOEx4N0Gh08A/pz5/s/Al0o6KAEghPB2COG/me9X4n/p7IqeTyoEtyrzdovMVwCOAP6aOa7nkxAz6wwcA/wx897Qs0m71PzZVu3B2q7AmznvF2aOSbrsGEJ4GzxgADolPJ6qZ2bdgIOAZ9DzSY3MMttzwBJgEjAXWBFCWJc5RX/GJeda4CdAfeZ9B/Rs0iQAD5vZTDM7M3MsNX+2tU7qxilheY5pe6xIE8ysLXAfcE4I4X2fIJA0CCGsBw40s+2B+4Ge+U4r7ajEzI4FloQQZprZYdnDeU7Vs0lO/xDCIjPrBEwys5eTHlCuap9ZWwh0yXnfGViU0FikcYvNbGeAzOuShMdTtcxsCzxQGx9C+FvmsJ5PyoQQVgCP4LmF25tZ9h/m+jMuGf2B481sHp5ucwQ+06ZnkxIhhEWZ1yX4P3T6kqI/26o9WKsFumd25GwJDAEmJDwm2dQE4LTM96cBf09wLFUrk2PzJ2B2COHqnB/p+aSAme2QmVHDzLYGvojnFU4Fvpo5Tc8nASGEC0MInUMI3fC/Z/4TQhiGnk0qmNknzWyb7PfAQOAlUvRnW9UXxTWzo/F/4dQA40IIoxIeUlUzs7uAw4COwGLgYuAB4B5gN2ABcFIIoeEmBImZmX0OeBx4kQ15NxfheWt6Pgkzs154EnQN/g/xe0IIl5rZHvhsTnvgWeCUEMLq5EZa3TLLoOeFEI7Vs0mHzHO4P/O2NXBnCGGUmXUgJX+2VX2wJiIiIpJm1b4MKiIiIpJqCtZEREREUkzBmoiIiEiKKVgTERERSTEFayIiIiIppmBNRCqama03s+dyviJrxmxm3czspaiuJyKST7W3mxKRyvdRCOHApAchItJcmlkTkapkZvPM7Eozm5752itzvKuZTTGzFzKvu2WO72hm95vZ85mvfplL1ZjZzWZWZ2YPZ7oHYGZnmdmszHXuTujXFJEKoGBNRCrd1g2WQU/O+dn7IYS+wA14JxMy398WQugFjAd+lzn+O+DREMKngN5AXeZ4d2BMCGE/YAVwYub4BcBBmet8N65fTkQqnzoYiEhFM7NVIYS2eY7PA44IIbyeaVD/fyGEDma2DNg5hLA2c/ztEEJHM1sKdM5tB2Rm3YBJIYTumffnA1uEEC43s4eAVXi7tAdCCKti/lVFpEJpZk1Eqllo5PvGzsknt5fjejbkAh8DjAE+Dcw0M+UIi0izKFgTkWp2cs7rU5nvnwSGZL4fBjyR+X4K8D0AM6sxs20bu6iZtQK6hBCmAj8Btgc2md0TESmE/qUnIpVuazN7Luf9QyGEbPmOrczsGfwfrkMzx84CxpnZj4GlwPDM8bOBm8zsDHwG7XvA243cswa4w8y2Awy4JoSwIrLfSESqinLWRKQqZXLW+oQQliU9FhGRpmgZVERERCTFNLMmIiIikmKaWRMRERFJMQVrIiIiIimmYE1EREQkxRSsiYiIiKSYgjURERGRFFOwJiIiIpJi/w9KQ2xrW/sMFQAAAABJRU5ErkJggg==\n",
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
    "train_acc_combined = history_combined.history['acc']\n",
    "val_acc_combined = history_combined.history['val_acc']\n",
    "epochs_combined = range(1, len(train_acc_combined) + 1)\n",
    "\n",
    "plt.plot(epochs_combined, train_acc_combined, 'bo', label='Training acc with Dropout and Data Augmentation')\n",
    "plt.plot(epochs_combined, val_acc_combined, 'b', label='Validation acc with Dropout and Data Augmentation')\n",
    "plt.title('Training and validation accuracy with Dropout and Data Augmentation')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "d6b46aa2",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save('model_dropout_augmentation.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff89fcfa",
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
