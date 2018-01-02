import pickle
from sklearn.linear_model import LinearRegression

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
a = input("Enter val")
x = loaded_model.predict(int(a))
print(x)




