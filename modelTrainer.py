import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle


df = pd.read_csv('dataset/coordinates_test.csv')

X = df.drop('class', axis=1)
y = df['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)


pipelines = {
    'lr': make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=1000)),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}


fit_models = {}
for algo, pipeline in pipelines.items():
    print("a")
    model = pipeline.fit(X_train, y_train)
    print("b")
    fit_models[algo] = model
    print("c")

print("done")

#Loss function & accuracy test
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    print(algo, accuracy_score(y_test, yhat))

fit_models['rf'].predict(X_test)

with open('model/body_language_demo.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)
