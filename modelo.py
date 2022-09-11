import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import cv2
import os
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Flatten 
from keras import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dropout
from sklearn.model_selection import train_test_split


CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class modeloPredicao:

    def __init__(self) -> None:

        self.data = pd.read_csv(r"C:\Users\akira\TCC\challenge - deep learn\csv\fer2013.csv")
        self.helper = Helper()
        self.train_data_dir = r"C:\Users\akira\TCC\content\data\train"
        self.validation_data_dir =  r"C:\Users\akira\TCC\content\data\test"


    def execute(self):

        self.helper.plot_data()

        print('-' * 60)

        pprint(self.data.isna().value_counts())

        print('-' * 60)

        self.data.info()

        print('-' * 60)

        self.criaDataset()

        print('-' * 60)

        processamento = self.helper.processaImagens()

    def processamentoDados(self):
        
        data = self.data

        data["emotion"].value_counts().reset_index(drop=True, inplace=True)

        X = data.drop("emotion", axis=1)
        y = data["emotion"]

        df = pd.concat([X, y], axis=1)

        print(df["emotion"].value_counts())

        return df

    def criaDataset(self):
        
        classes = ['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

        df = self.processamentoDados()

        df['pixels'] = df["pixels"].apply(self.helper.pixels_to_array)

        data_train = df[df["Usage"] == "Training"]
        data_test1 = df[df["Usage"] == "PublicTest"]
        data_test2 = df[df["Usage"] == "PrivateTest"]

        data_test = pd.concat([data_test1, data_test2])

        X_train = self.helper.image_reshape(data_train["pixels"])

        X_test = self.helper.image_reshape(data_test["pixels"])
        y_train = data_train["emotion"]
        y_test = data_test["emotion"]

        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        self.helper.put_in_dir(X_train, X_test, y_train, y_test, classes)
        

class Helper:

    def __init__(self) -> None:

        self.data = pd.read_csv(r"C:\Users\akira\TCC\challenge - deep learn\csv\fer2013.csv")
        self.train_data_dir = r"C:\Users\akira\TCC\content\data\train"
        self.validation_data_dir =  r"C:\Users\akira\TCC\content\data\test"


    def plot_data(self):
  
        values = self.data["emotion"].value_counts().sort_index(ascending=True)
        colors = ["lightgreen", "blue", "lightblue", "pink", "orange", "yellow", "purple"]

        plt.figure(figsize=[12, 5])
        
        plt.bar(x=CLASSES, height=values, color=colors, edgecolor='black')

        plt.xlabel("Emoções")
        plt.ylabel("Quantidade")
        plt.title("Graficos de emoções")
        plt.show()
  
    def pixels_to_array(self,pixels):

        #Transformando em float

        array = np.array(pixels.split(),'float64')

        return array

    def image_reshape(self, data):

        #Ajustando o shape

        image = np.reshape(data.to_list(),(data.shape[0],48,48,1))
        image = np.repeat(image, 3, -1)

        return image

    
    def put_in_dir(self, X_train, X_test, y_train, y_test, classes):
        """
            Função que me tras as imagens para os diretorios corretos

        """
        print('-' * 60)
        pprint("Verificando diretorios...")

        for label in tqdm(range(len(classes))):
            os.makedirs(r"C:\Users\akira\TCC\content\data\train\\" + classes[label], exist_ok=True)
            os.makedirs(r"C:\Users\akira\TCC\content\data\test\\" + classes[label], exist_ok=True)

        print('-' * 60)
        pprint("Adicionando imagens de treino no diretorio...")

        for i in tqdm(range(len(X_train))):
            emotion = classes[y_train[i]]
            cv2.imwrite(f"C:\\Users\\akira\\TCC\\content\\data\\train\\{emotion}\\{emotion}{i}.png", X_train[i])

        print('-' * 60)
        pprint("Adicionando imagens de teste no diretorio...")
        
        for j in tqdm(range(len(X_test))):
            emotion = classes[y_test[j]]
            cv2.imwrite(f"C:\\Users\\akira\\TCC\\content\\data\\test\\{emotion}\\{emotion}{j}.png", X_test[j])


    def processaImagens(self):

        from keras.preprocessing.image import ImageDataGenerator

        train_data_dir = self.train_data_dir
        validation_data_dir = self.validation_data_dir

        train_datagen = ImageDataGenerator(
                                            rescale=1./255,
                                            rotation_range=30,
                                            shear_range=0.3,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_data = train_datagen.flow_from_directory(directory= train_data_dir, 
                                               target_size=(224,224), 
                                               batch_size=32,
                                  )

        print(train_data.class_indices)

        val_data = validation_datagen.flow_from_directory(directory= validation_data_dir, 
                                           target_size=(224,224), 
                                           batch_size=32,
                                  )

        pprint("Executando compilador...")

        self.checkImagem(train_data)
        self.criandoModelo(train_datagen, train_data_dir, validation_datagen, validation_data_dir)


    def criandoModelo(self, train_datagen, train_data_dir, validation_datagen, validation_data_dir):

        train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(48, 48),
					batch_size=32,
					class_mode='categorical',
					shuffle=True)

        validation_generator = validation_datagen.flow_from_directory(
                                    validation_data_dir,
                                    color_mode='grayscale',
                                    target_size=(48, 48),
                                    batch_size=32,
                                    class_mode='categorical',
                                    shuffle=True)

        self.checkSummary(train_generator)
        self.compilador(train_generator, validation_generator)


    def checkImagem(self, train_data):
        
        t_img , label = train_data.next()

        """
        Verificando cada imagem nos diretorios de emoções
        """
        count = 0
        for im, l in zip(t_img,label) :
            plt.imshow(im)
            plt.title(im.shape)
            plt.axis = False
            plt.show()
            
            count += 1
            if count == 10:
                break


    def checkSummary(self, train_generator):

        img, label = train_generator.__next__()

        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        self.modelo = model



    def compilador(self, train_generator, validation_generator):
        
        ## Percorrendo as pastas que contém as imagens de treino e teste

        num_train_imgs = 0

        for root, dirs, files in os.walk(self.train_data_dir):
            num_train_imgs += len(files)
            
        num_test_imgs = 0

        for root, dirs, files in os.walk(self.validation_data_dir):
            num_test_imgs += len(files)

        print('-' * 60)
        print(num_train_imgs)
        print(num_test_imgs)

        epochs=30

        history= self.modelo.fit(train_generator,
                        steps_per_epoch=num_train_imgs//32,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=num_test_imgs//32)

        self.modelo.save('model_file.h5')



if __name__ == '__main__':
    ja = modeloPredicao()
    ja.execute()