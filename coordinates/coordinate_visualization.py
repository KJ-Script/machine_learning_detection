import pandas as pd

df = pd.read_csv('../dataset/coordinates.csv')
head = df.head()
tail = df.tail()

frame = df[df['class']=='Sad']

X = df.drop('class', axis=1)
y = df['class']


print("head", head)
print("tail", tail)
print("frame", frame)
print("X", X)
print("Y", y)
