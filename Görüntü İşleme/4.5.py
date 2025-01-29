{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "654166c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "from tensorflow.keras.models import load_model\n",
    "import numpy as np\n",
    "import time  # Zamanı ölçmek için kütüphane ekleyin"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "2d50775d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Eğittiğiniz modeli yükleyin\n",
    "model_path = 'model_training_validation.h5'\n",
    "model = load_model(model_path)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "1f356c26",
   "metadata": {},
   "outputs": [
    {
     "ename": "error",
     "evalue": "OpenCV(3.4.1) C:\\Miniconda3\\conda-bld\\opencv-suite_1533128839831\\work\\modules\\imgproc\\src\\resize.cpp:4044: error: (-215) ssize.width > 0 && ssize.height > 0 in function cv::resize\n",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31merror\u001b[0m                                     Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-12-d3524f5148a1>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m     16\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     17\u001b[0m         \u001b[1;31m# Görüntüyü model için uygun formata getirin\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 18\u001b[1;33m         \u001b[0mresized_frame\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcv2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mresize\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mframe\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m(\u001b[0m\u001b[1;36m48\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m48\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     19\u001b[0m         \u001b[0mimg_array\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mexpand_dims\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mresized_frame\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m0\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;33m/\u001b[0m \u001b[1;36m255.0\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     20\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31merror\u001b[0m: OpenCV(3.4.1) C:\\Miniconda3\\conda-bld\\opencv-suite_1533128839831\\work\\modules\\imgproc\\src\\resize.cpp:4044: error: (-215) ssize.width > 0 && ssize.height > 0 in function cv::resize\n"
     ]
    }
   ],
   "source": [
    "# Webcam'den görüntü almak için bir capture objesi oluşturun\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "# FPS (kare hızı) sınırlayıcıları\n",
    "fps_limit = 5  # Her saniyede en fazla 5 kare işleyeceğiz\n",
    "interval = 1.0 / fps_limit  # Güncelleme aralığı hesaplanıyor\n",
    "\n",
    "last_time = time.time()  # Başlangıç zamanını alın\n",
    "\n",
    "while True:\n",
    "    ret, frame = cap.read()  # Webcam'den bir frame alın\n",
    "\n",
    "    # Her saniyede belirli bir kare hızı ile tahminleri güncelleyin\n",
    "    if time.time() - last_time >= interval:\n",
    "        last_time = time.time()  # Zamanı güncelleyin\n",
    "\n",
    "        # Görüntüyü model için uygun formata getirin\n",
    "        resized_frame = cv2.resize(frame, (48, 48))\n",
    "        img_array = np.expand_dims(resized_frame, axis=0) / 255.0\n",
    "\n",
    "        # Eğitilen modeli kullanarak tahmin yapın\n",
    "        prediction = model.predict(img_array)\n",
    "\n",
    "        # Tahmin sonucunu işleyin\n",
    "        prob_happy = prediction[0][0] * 100\n",
    "        prob_sad = (1 - prediction[0][0]) * 100\n",
    "\n",
    "        if prediction[0][0] > 0.5:\n",
    "            label = \"Happy: {:.2f}% | Sad: {:.2f}%\".format(prob_happy, prob_sad)\n",
    "        else:\n",
    "            label = \"Happy: {:.2f}% | Sad: {:.2f}%\".format(prob_happy, prob_sad)\n",
    "\n",
    "        # Etiketi ve çerçeveyi görselleştirin\n",
    "        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)\n",
    "        cv2.imshow('Emotion Detection', frame)\n",
    "\n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):  # Çıkış için 'q' tuşuna basın\n",
    "        break\n",
    "\n",
    "# Döngüden çıkınca işlemleri serbest bırakın ve pencereyi kapatın\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "902953bd",
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
