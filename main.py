import cv2
import numpy as np
from keras.models import load_model

labels_dict={0:'Bravo',1:'Enojado', 2:'Medo', 3:'Feliz',4:'Neutro',5:'Triste',6:'Supreso'}

class verificandoModelo:

    def __init__(self) -> None:
        self.model = load_model('model_file.h5')
        self.cap = cv2.VideoCapture(0)
        self.face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    def execute(self):
        
        video = self.cap
        detector =self.face
        modelo = self.model

        while True:
            ret,frame=video.read()
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces= detector .detectMultiScale(gray, 1.3, 3)

            for x,y,w,h in faces:
                sub_face_img=gray[y:y+h, x:x+w]
                resized=cv2.resize(sub_face_img,(48,48))
                normalize=resized/255.0
                reshaped=np.reshape(normalize, (1, 48, 48, 1))
                result=modelo.predict(reshaped)
                label=np.argmax(result, axis=1)[0]

                print(label)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
                cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
                cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                
            cv2.imshow("Captura de tela",frame)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ja = verificandoModelo()
    ja.execute()