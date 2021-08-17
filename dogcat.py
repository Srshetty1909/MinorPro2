from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
dogcat = Sequential()
dogcat.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))
dogcat.add(MaxPooling2D(pool_size = (2, 2)))
dogcat.add(Conv2D(32, (3, 3), activation = 'relu'))
dogcat.add(MaxPooling2D(pool_size = (2, 2)))
dogcat.add(Flatten())
dogcat.add(Dense(units = 128, activation = 'relu'))
dogcat.add(Dense(units = 1, activation = 'sigmoid'))
dogcat.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('./dogs-vs-cats/train',
                                                 target_size = (50, 50),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
dogcat.fit_generator(training_set,
                         steps_per_epoch = 4000,
                         epochs = 25,
                         validation_steps = 2000)
dogcat_json = dogcat.to_json()
with open("./dogcat.json","w") as json_file:
  json_file.write(dogcat_json)
dogcat.save_weights("./dogcat.h5")

print("Classifier trained Successfully!")
