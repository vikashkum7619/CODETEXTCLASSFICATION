import pickle
import joblib
from keras.models import Model



def loading_pickle(model_path):
    vectorizer = pickle.load(open(model_path+"vectorizer.pickle",'rb'))
    model = pickle.load(open(model_path+"model.pickle",'rb'))

    return vectorizer, model


def loading_joblibPickle(model_path):
    vectorizer = joblib.load(model_path+"vectorizer.sav")
    model = joblib.load(model_path+"model.sav")

    return vectorizer, model


def predict(model, vectorizer, text):
    pred = model.predict(vectorizer.transform([text]))[0]

  

    print("predicted class:", pred) 
    print(pred)


    return pred



if __name__ == '__main__':
    Project_path = "MULTICLASS"
    model_path = Project_path + "/Multiclasstext/models/"


    text = "text here"


            

    # TEST using PICKLE
    vectorizer, model  = loading_pickle(model_path)
    predict(model, vectorizer, text)

    vectorizer1, model1 = loading_joblibPickle(model_path)
    predict(model1, vectorizer1, text)
