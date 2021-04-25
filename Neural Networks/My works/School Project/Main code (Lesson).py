import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras import utils
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
import matplotlib.pyplot as plt

!wget https://github.com/Horea94/Fruit-Images-Dataset/archive/master.zip -O master.zip
!unzip master.zip

train_dataset = image_dataset_from_directory('Fruit-Images-Dataset-master/Training', 
                                              subset='training', 
                                              seed=42, validation_split=0.1, 
                                              batch_size=256, 
                                              image_size=(100, 100))
validation_dataset = image_dataset_from_directory('Fruit-Images-Dataset-master/Training', 
                                                    subset='validation', 
                                                    seed=42, validation_split=0.1, 
                                                    batch_size=256, 
                                                    image_size=(100, 100))
test_dataset = image_dataset_from_directory('Fruit-Images-Dataset-master/Test', 
                                              batch_size=256, 
                                              image_size=(100, 100))

class_names = train_dataset.class_names

model = Sequential()
#Часть для свертки
model.add(Conv2D(16, (5, 5), padding='same', input_shape=(100, 100, 3), activation='relu'))  #same значит входящие файлы имеют одинаковыю ширину и высоту, размер картинки 100 на 100
model.add(MaxPooling2D(pool_size=(2,2))) #pool означает max pooling (выбор максимума из значений) из области 2 на 2
model.add(Conv2D(32, (5, 5), padding='same', activation='relu')) #relu функция активации которая возвращает x при x>0, 32 фильтра, которые применяются к изображению
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#Часть для классификации
model.add(Flatten()) #Входные слои
model.add(Dense(1024, activation='relu')) #Скрытый слой (1024 нейрона)
model.add(Dropout(0.2)) #Выключиние 0.2 части нейронов, для лучшей обучаемости
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(131, activation='softmax')) #131 выходной нейрон, так как 131 класс фруктов, активация максимального

model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy') 
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=5, verbose=1) 

scores = model.evaluate(test_dataset, verbose=1) #точность предсказаний
print("Доля верных ответов в процентах: ", scores[1]*100)

plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Ошибка')
plt.legend()
plt.show()

model.save("fruits_360_model.h5")
files.download("fruits_360_model.h5")
