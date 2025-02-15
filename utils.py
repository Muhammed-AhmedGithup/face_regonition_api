import cv2
import wavelet
import numpy as np
import joblib
import json
import warnings
import base64
warnings.filterwarnings('ignore')

model=None
classes_name_to_number={}
class_number_to_name={}
def classify_image(image_bs4,image_path=None):
    images=cropped_2eys(image_base64=image_bs4,image_path=image_path)
    final=[]
    for image in images:
        image_face_color=cv2.resize(image,(50,50))
        image_wv=wavelet.w2d(img=image,mode='db1',level=5)
        image_wv=cv2.resize(image_wv,(50,50))
        combine=np.vstack((image_face_color.reshape(50*50*3,1),image_wv.reshape(50*50,1)))
        combine=np.array(combine).reshape(1,10000)
        final.append({
            'class': class_number_to_name[model.predict(combine)[0]],
            'class_probability': np.around(model.predict_proba(combine)*100,2).tolist()[0],
            'class_dictionary': classes_name_to_number
        })
    return final


def cropped_2eys(image_path,image_base64):
    face_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eys_cascade=cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    if image_base64 is None:
        image=cv2.imread(image_path)
    else:
        image=get_cv2_image_from_base64_string(image_base64)
        
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_image,1.3,5)
    
    cropped_faces=[]
    for(x,y,w,h) in faces:
        gray_p=gray_image[y:y+h,x:x+w]
        face_color=image[y:y+h,x:x+w]
        eyes=eys_cascade.detectMultiScale(gray_p)
        if len(eyes)>=2:
            cropped_faces.append(face_color)
    return cropped_faces
        
def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
def load_artifactes():
    global model
    global classes_name_to_number
    global class_number_to_name
    
    with open('./artifactes/model.pkl','rb') as f:
        model=joblib.load(f)
        
    with open('./artifactes/classes_dict.json','r') as f:
        classes_name_to_number=json.load(f) 
        class_number_to_name={v:k for k,v in classes_name_to_number.items()}
        

if __name__ == '__main__':  
    load_artifactes()