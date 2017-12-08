import pickle
from sklearn.linear_model import LinearRegression

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
x = loaded_model.predict(150)
print(x)




