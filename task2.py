import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

train_path = 'D:/Internships/codesoft/task 2/fraudTrain.csv'
test_path = 'D:/Internships/codesoft/task 2/fraudTest.csv'


def load_data(train_path: str, test_path: str):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def preprocessing(df: pd.core.frame.DataFrame, target_column: str):
    df['age'] = pd.to_datetime('today').year - pd.to_datetime(df['dob']).dt.year
    df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
    df['day'] = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek
    df['month'] = pd.to_datetime(df['trans_date_trans_time']).dt.month

    columns_list = ['category', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'age', 'hour', 'day',
                    'month', target_column]

    df = df[columns_list]
    df = pd.get_dummies(df, drop_first=True)

    y_df = df[target_column].values
    x_df = df.drop(target_column, axis='columns').values
    return x_df, y_df


def build_and_evaluate_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    LR_model = LogisticRegression()
    LR_model.fit(X_train_resampled, y_train_resampled)

    LR_prediction = LR_model.predict(X_test)

    LR_classification_report = classification_report(y_test, LR_prediction)
    LR_confusion_matrix = confusion_matrix(y_test, LR_prediction)
    LR_accuracy = accuracy_score(y_test, LR_prediction)

    return LR_classification_report, LR_confusion_matrix, LR_accuracy


def main():
    train_data, test_data = load_data(train_path, test_path)

    # Preprocess the data
    X_train, y_train = preprocessing(train_data, 'is_fraud')
    X_test, y_test = preprocessing(test_data, 'is_fraud')

    # Build and evaluate the model
    report, confusion_matrix, accuracy = build_and_evaluate_model(X_train, y_train, X_test, y_test)
    print(f'Logistic Regression Classification Report:\n{report}')
    print(f'Logistic Regression Confusion matrix:\n{confusion_matrix}')
    print(f'Logistic Regression Accuracy:\n{accuracy}')

if __name__ == "__main__":
    main()
