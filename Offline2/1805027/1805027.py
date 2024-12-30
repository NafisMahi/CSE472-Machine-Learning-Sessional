import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataset_preprocess as dp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessTeclo():
    #----------------------Importing the dataset----------------------#
    dataset = pd.read_csv('dataset1/teclo.csv')

    dataset.iloc[:, 19] = dataset.iloc[:, 19].replace(' ', np.nan)
    dataset.iloc[:, 19] = pd.to_numeric(dataset.iloc[:, 19], errors='coerce')
    col_mean = dataset.iloc[:, 19].mean()
    dataset.iloc[:, 19].fillna(col_mean, inplace=True)

    #----------------------Train Test split---------------------------#
    x = dataset.drop('Churn',axis=1)
    y = pd.DataFrame(dataset['Churn'].values)
    y = y.values.ravel()

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    y_test = np.where(y_test == 'Yes', 1, -1)
    y_train = np.where(y_train == 'Yes', 1, 0)


    #----------------------Feature Scaling-----------------------------#
    sc = StandardScaler()
    columns_to_scale = ['tenure','MonthlyCharges','TotalCharges']
    x_train[columns_to_scale] = sc.fit_transform(x_train[columns_to_scale])
    x_test[columns_to_scale] = sc.transform(x_test[columns_to_scale])


    #----------------------Encoding Categorical Data-------------------#
    x_train = x_train.drop('customerID',axis=1)
    x_test = x_test.drop('customerID',axis=1)

    non_numeric_columns = x_train.select_dtypes(exclude=['int64', 'float64']).columns

    x_train = pd.get_dummies(x_train, columns=non_numeric_columns, dtype=int)
    x_test = pd.get_dummies(x_test, columns=non_numeric_columns, dtype=int)

    return x_train.values,x_test.values,y_train,y_test

def preprocessAdult():
    #----------------------Importing the dataset----------------------#
    train_dataset = pd.read_csv('dataset2/adult.data', header=None, na_values=' ?')
    test_dataset = pd.read_csv('dataset2/adult.test', header=None, na_values=' ?')

    #----------------------Handle missing data---------------------------#
    for col in train_dataset.columns:
        if train_dataset[col].dtype == 'float64' or train_dataset[col].dtype == 'int64':
            train_dataset[col].fillna(train_dataset[col].mean(), inplace=True)

        else:
            train_dataset[col].fillna(train_dataset[col].mode()[0], inplace=True)
        
    for col in test_dataset.columns:
        if test_dataset[col].dtype == 'float64' or test_dataset[col].dtype == 'int64':
            test_dataset[col].fillna(test_dataset[col].mean(), inplace=True)
            
        else:
            test_dataset[col].fillna(test_dataset[col].mode()[0], inplace=True)


    #----------------------X,y split-----------------------------#
    x_train = train_dataset.drop(train_dataset.columns[-1], axis=1)
    y_train = train_dataset[train_dataset.columns[-1]].values
    x_test = test_dataset.drop(test_dataset.columns[-1], axis=1)
    y_test = test_dataset[test_dataset.columns[-1]].values
    y_train = np.where(y_train == ' >50K', 1, 0)
    y_test = np.where(y_test == ' >50K.', 1, -1)


    #----------------------Feature Scaling-------------------#
    sc = StandardScaler()

    columns_to_scale = [0,2,4,10,11,12]

    x_train[columns_to_scale] = sc.fit_transform(x_train[columns_to_scale])
    x_test[columns_to_scale] = sc.transform(x_test[columns_to_scale])

    #----------------------Encoding Categorical Data-------------------#
    non_numeric_columns = x_train.select_dtypes(exclude=['int64', 'float64']).columns

    x_train = pd.get_dummies(x_train,columns=[1,3,5,6,7,8,9,13],dtype=int)
    x_test = pd.get_dummies(x_test,columns=[1,3,5,6,7,8,9,13],dtype=int)

    #------------------------Align train and test data------------------#
    missing_cols = set(x_train.columns) - set(x_test.columns)

    for col in missing_cols:
        x_test[col] = 0

    # Reorder x_test columns to match the order in x_train
    x_test = x_test[x_train.columns]

    x_train = x_train.values
    x_test = x_test.values


    return x_train,x_test,y_train,y_test

def preprocessCreditcard():
    #----------------------Importing the dataset----------------------#
    dataset = pd.read_csv('dataset3/creditcard.csv')
    # Separate the classes
    class_1 = dataset[dataset['Class'] == 1]
    class_0 = dataset[dataset['Class'] == 0]

    # Determine how many more examples are needed to reach 20,000
    additional_samples_needed = 20000 - len(class_1)

    # Sample additional examples from class_0
    class_0_sampled = class_0.sample(n=additional_samples_needed, random_state=42)

    # Combine the datasets to make the new balanced dataset
    balanced_dataset = pd.concat([class_1, class_0_sampled])

    # Shuffle the dataset
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    dataset = balanced_dataset

    # print(balanced_dataset.shape)

    #----------------------X,y split-----------------------------#
    x = dataset.drop('Class',axis=1)
    y = pd.DataFrame(dataset['Class'].values)
    y = y.values.ravel()

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
    
    y_test = np.where(y_test == 1, 1, -1)
    number_of_ones = np.sum(y_test == 1)

    #----------------------Feature Scaling-------------------#
    sc = StandardScaler()
    columns_to_scale = ['Time','Amount']
    x_train[columns_to_scale] = sc.fit_transform(x_train[columns_to_scale])
    x_test[columns_to_scale] = sc.transform(x_test[columns_to_scale])

    return x_train.values,x_test.values,y_train,y_test


def information_gain_function(x_train, y_train, k):
    bestFeatures = SelectKBest(score_func=mutual_info_classif,k=k)
    selector = bestFeatures.fit(x_train, y_train)
    x_train = selector.transform(x_train)
    selected_indices = np.where(selector.get_support())[0]
   
    return x_train, selected_indices

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    gradient = np.dot(X.T, (predictions - y)) / m
    return gradient

def compute_cost(X, y, weights):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights))
    cost = (1 / m) * (np.dot(-y.T, np.log(predictions)) - np.dot((1 - y).T, np.log(1 - predictions)))
    return cost

def gradient_descent(X, y, num_features, learning_rate=0.01, max_iter=1000, threshold=0):
    X = np.insert(X, 0, 1, axis=1) 
    weights = np.random.rand(num_features + 1) * 0.01
    for i in range(max_iter):
        gradient = compute_gradient(X, y, weights)
        weights -= learning_rate * gradient
        
        cost = compute_cost(X, y, weights)

        # Early stopping condition
        
        if cost < threshold:
            print(f"Early stopping at iteration {i}")
            break

    return weights

def logistic_regression(x_train, y_train, k, learning_rate, max_iter, threshold):

    x_train_selected, selected_indices = information_gain_function(x_train, y_train, k)

    weights = gradient_descent(x_train_selected, y_train, k, learning_rate, max_iter, threshold)
    
    return weights, selected_indices

def predict(X, weights, selected_indices):
   
    x_test_selected = X[:, selected_indices]
    x_test_selected = np.insert(x_test_selected, 0, 1, axis=1)
    
    predictions = sigmoid(np.dot(x_test_selected, weights)) 
    predictions = np.where(predictions >= 0.5, 1, 0)
    return predictions

def adaboost(X, y, num_weak_learners, k, learning_rate, max_iter, threshold):
    N = len(y)
    w = np.ones(N) / N
    models = []
    alphas = []

    random_seed = 42
    np.random.seed(random_seed)


    for _ in range(num_weak_learners):

        # Resample the data points based on their weights
        sampled_indices = np.random.choice(N, size=N, p=w)
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]

        # Train the weak learner
        weights, selected_indices = logistic_regression(X_sampled, y_sampled, k, learning_rate, max_iter, threshold)
        model = (weights, selected_indices)
        models.append(model)
        
        # Get weak learner predictions
        predictions = predict(X, weights, selected_indices)
        
        # Error weighted by sample weights
        indicator = np.not_equal(predictions, y)
        error = np.sum(w[indicator])
        
        if error == 0:
            break
        
        if error > 0.5:
            continue
        
        alpha = 0.5 * np.log((1 - error) / error)
        alphas.append(alpha)
        
        # Update sample weights
        for j in range(len(w)):
            if predictions[j] == y[j]:
                w[j] = w[j] * error / (1 - error)
        w /= np.sum(w)

    return models, alphas


def adaPredict(X, models, alphas):
    N = len(X)
    predictions = np.zeros(N)
    for alpha, (weights, selected_indices) in zip(alphas, models):

        weak_predictions = predict(X,weights, selected_indices)
        weak_predictions = np.where(weak_predictions >= 0.5, 1, -1)
        
        predictions += alpha * weak_predictions
    
    predictions = np.where(predictions >= 0, 1, -1)
    return predictions


def performanceMetrics(k,num_weak_learners,learning_rate,max_iter,threshold):
    x_train, x_test, y_train, y_test = preprocessTeclo()

    models, alphas = adaboost(x_train, y_train, num_weak_learners, k, learning_rate, max_iter, threshold)

    y_train = np.where(y_train == 1, 1, -1)

    # Make predictions
    y_pred = adaPredict(x_train, models, alphas)

    # Compute accuracy
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    confusion = confusion_matrix(y_train, y_pred)

    TN, FP, FN, TP = confusion.ravel()

    # Calculate Specificity
    specificity = TN / (TN + FP)

    # Calculate False Discovery Rate
    fdr = FP / (FP + TP)

    # # Make predictions
    # y_pred = adaPredict(x_test, models, alphas)

    # # Compute accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    # confusion = confusion_matrix(y_test, y_pred)

    # TN, FP, FN, TP = confusion.ravel()

    # # Calculate Specificity
    # specificity = TN / (TN + FP)

    # # Calculate False Discovery Rate
    # fdr = FP / (FP + TP)

    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Accuracy", f"{accuracy:.3f}"])
    table.add_row(["Precision", f"{precision:.3f}"])
    table.add_row(["Recall", f"{recall:.3f}"])
    table.add_row(["Specificity", f"{specificity:.3f}"])
    table.add_row(["False Discovery Rate", f"{fdr:.3f}"])
    table.add_row(["F1 Score", f"{f1:.3f}"])
    table.add_row(["Confusion Matrix", f"{confusion}"])
    print(table)


def runAll():
    values = [1, 5, 10, 15, 20]
    top_k = 10
    learning_rate = 0.01
    epochs = 1000
    threshold = 0.0

    print("Results for normal logistic regression:\n")
    performanceMetrics(10, 1, 0.01, 5000, 0)

    for k in values:
        print(f"Results for AdaBoost with {k} rounds:\n")
        performanceMetrics(top_k, k, learning_rate, epochs, threshold)

runAll()