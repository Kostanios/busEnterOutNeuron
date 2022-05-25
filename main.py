from tensorflow.keras.models import Sequential      # Подлючение класса создания модели Sequential
from tensorflow.keras.layers import Dense           # Подключение класса Dense (полносвязного слоя)
from tensorflow.keras import utils                  # Утилиты для to_categorical
import numpy as np                                  # Подключение библиотеки numpy
from tensorflow.keras.preprocessing.image import load_img
import gdown
gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l4/bus.zip', None, quiet=True)
%matplotlib inline
!unzip -q bus.zip

# Создание полотна из десяти графиков
# figEntry, axsEntry = plt.subplots(1, 10, figsize=(25, 5))
# figOut, axsOut = plt.subplots(1, 10, figsize=(25, 5))

x = arr = np.empty(( 0, 56, 56, 3));
y = np.empty((0,), dtype='int');

def getPath (str):
  return '0' * (5 - len(str)) + str
# Проход и отрисовка по всем классам

test_size = 5000
# 0 - входящий 1 - выходящий

def useImageHandler (path):
  img = load_img(path, target_size=(56, 56))
  return np.array(1 - np.array(img) / 255)

for i in range(int(test_size/2)):
    # axsEntry[i].imshow(imgEntry, cmap = 'gray')
    # axsOut[i].imshow(imgOut, cmap = 'gray')
    x = np.append(x, [useImageHandler('Входящий/' + getPath(str(i)) + '.jpg')], axis=0)
    y = np.append(y, 0)
    x = np.append(x, [useImageHandler('Выходящий/' + getPath(str(i)) + '.jpg')], axis=0)
    y = np.append(y, 1)

# Вывод изображений
# plt.show()
print(np.shape(x))
print(len(x))
print(np.shape(y))
print(len(y))

r_x = np.reshape(x, (test_size, 9408))
t_r_x = r_x.astype('float32')

y = utils.to_categorical(y, 2)

train_size = int(test_size*0.9)
xtrain = t_r_x[0:test_size]
xtest = t_r_x[test_size:]

ytrain = y[0:test_size]
ytest = y[test_size:]


# Создание сети прямого распространения
model = Sequential() 
# Добавление полносвязного слоя на 8000 нейронов с relu-активацией
model.add(Dense(10000, input_dim=9408, activation="relu")) 
# Добавление полносвязного слоя на 6000 нейронов с relu-активацией
model.add(Dense(6000, activation="relu")) 
# Добавление полносвязного слоя на 2000 нейронов с relu-активацией
model.add(Dense(2000, activation="relu")) 
# Добавление полносвязного слоя на 1000 нейронов с relu-активацией
model.add(Dense(1000, activation="relu"))
# Добавление полносвязного слоя на 2 нейронов с softmax-активацией
model.add(Dense(2, activation="softmax")) 

# Компиляция модель
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 

# Вывод структуры модели
print(model.summary())

model.fit(xtrain, ytrain, batch_size=3136, epochs=200, verbose=1)

model.save_weights('model.h5')          # Сохранение весов модели
model.load_weights('model.h5')          # Загрузка весов модели