import pandas as pd 
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('creditcard.csv')

mlflow.set_experiment('Credit Card Monitoring')

mlflow.sklearn.autolog(log_models=True)

df = df[:10000]

x, y =  df.drop('Class', axis=1), df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, stratify=y)

scoring = {"precision": "precision",
           "accuracy": "accuracy"
           
           }

params = {
    "n_estimators": [20, 50, 100],
    "max_depth": [5, 10, 15, 20]
    
    }

grid_search = GridSearchCV(
    estimator = RandomForestClassifier(),
    param_grid=params,
    scoring=scoring,
    cv=5,
    refit="precision"
)

with mlflow.start_run(run_name='test'):
    grid_search.fit(x_train, y_train)




