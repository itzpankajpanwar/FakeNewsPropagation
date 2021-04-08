import time

import sys
sys.path.insert(0,"/content/FakeNewsPropagation")


import matplotlib
import numpy as np
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

#from load_dataset import load_from_nx_graphs
#from construct_sample_features import get_nx_propagation_graphs
#from analysis_util import equal_samples
from centrality import get_degree_centrality,get_closness_centrality,get_betweenness_centrality,get_pagerank
from  construct_sample_features import get_TPNF_dataset, get_train_test_split, get_dataset_feature_names

matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_classifier_by_name(classifier_name):
    if classifier_name == "GaussianNB":
        return GaussianNB()
    elif classifier_name == "LogisticRegression":
        return LogisticRegression(solver='lbfgs')
    elif classifier_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier()
    elif classifier_name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=50)
    elif classifier_name == "SVM -linear kernel":
        return svm.SVC(kernel='linear')
    elif classifier_name == "ExtraTreesClassifier":
      return ExtraTreesClassifier(n_estimators=100)
    elif classifier_name == "AdaBoostClassifier":
      return AdaBoostClassifier(n_estimators=100, random_state=0)
    else:
      return KNeighborsClassifier(n_neighbors=3)


def train_model(classifier_name, X_train, X_test, y_train, y_test):
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_score_values = []

    for i in range(6):
        classifier_clone = get_classifier_by_name(classifier_name)
        classifier_clone.fit(X_train, y_train)

        predicted_output = classifier_clone.predict(X_test)
        accuracy, precision, recall, f1_score_val = get_metrics(y_test, predicted_output, one_hot_rep=False)

        accuracy_values.append(accuracy)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_score_values.append(f1_score_val)

    print_metrics(np.mean(accuracy_values), np.mean(precision_values), np.mean(recall_values), np.mean(f1_score_values))


def print_metrics(accuracy, precision, recall, f1_score_val):
    print("Accuracy : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("F1 : {}".format(f1_score_val))


def get_metrics(target, logits, one_hot_rep=True):
    """
    Two numpy one hot arrays
    :param target:
    :param logits:
    :return:
    """

    if one_hot_rep:
        label = np.argmax(target, axis=1)
        predict = np.argmax(logits, axis=1)
    else:
        label = target
        predict = logits

    accuracy = accuracy_score(label, predict)

    precision = precision_score(label, predict)
    recall = recall_score(label, predict)
    f1_score_val = f1_score(label, predict)

    return accuracy, precision, recall, f1_score_val


def apply_lstm_model(X_train, X_test, y_train, y_test):
    print("\n TRAIN AND TEST SHAPES FOR LSTM MODEL IS AS FOLLOWS : ")
    X_train = X_train[:,None,:]
    X_test = X_test[:,None,:]
    print("\n X TRAIN : ",X_train.shape)
    print("\n Y TRAIN ",y_train.shape)
    print("\n X TEST ",X_test.shape)
    print("\n Y TEST ",y_test.shape,"\n")

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    from keras.optimizers import Adam

    #Initializing the classifier Network
    classifier = Sequential()

    #Adding the input LSTM network layer
    classifier.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
    classifier.add(Dropout(0.2))

    #Adding a second LSTM network layer
    #classifier.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
    classifier.add(LSTM(128))

    #Adding a dense hidden layer
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dropout(0.2))

    #Adding the output layer
    classifier.add(Dense(10, activation='softmax'))

    #Compiling the network
    classifier.compile( loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=0.001, decay=1e-6),
                  metrics=['accuracy'] )

    #Fitting the data to the model
    classifier.fit(X_train,
            y_train,
              epochs=300,
              validation_data=(X_test, y_test))
    test_loss, test_acc = classifier.evaluate(X_test, y_test)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    

def get_basic_model_results(X_train, X_test, y_train, y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\n TRAIN AND TEST SHAPES ARE AS FOLLOWS")
    print("\n X TRAIN : ",X_train.shape)
    print("\n Y TRAIN ",y_train.shape)
    print("\n X TEST ",X_test.shape)
    print("\n Y TEST ",y_test.shape,"\n")

    

    classifiers = [GaussianNB(), LogisticRegression(), DecisionTreeClassifier(),
                   RandomForestClassifier(n_estimators=100),
                   svm.SVC(),ExtraTreesClassifier(n_estimators=100),KNeighborsClassifier(n_neighbors=3),AdaBoostClassifier(n_estimators=100, random_state=0)]
    classifier_names = ["GaussianNB", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier",
                        "SVM -linear kernel","ExtraTreesClassifier","KNeighborsClassifier","AdaBoostClassifier"]

    for idx in range(len(classifiers)):
        print("======={}=======".format(classifier_names[idx]))
        train_model(classifier_names[idx], X_train, X_test, y_train, y_test)
    
    #apply_lstm_model(X_train, X_test, y_train, y_test);
    

    
   


def get_classificaton_results_tpnf(data_dir, news_source, time_interval, use_cache=False):
    include_micro = True
    include_macro = True

    include_structural = True
    include_temporal = True
    include_linguistic = False
  
    dc=get_degree_centrality("/content/FakeNewsPropagation/data/nx_network_data/nx_network_data", news_source)
    cc=get_closness_centrality("/content/FakeNewsPropagation/data/nx_network_data/nx_network_data", news_source)
    pr=get_pagerank("/content/FakeNewsPropagation/data/nx_network_data/nx_network_data", news_source)
    

    sample_feature_arr = get_TPNF_dataset(data_dir, news_source, include_micro, include_macro, include_structural,
                                            include_temporal, include_linguistic, time_interval, use_cache=use_cache)
    dc = np.array(dc)
    cc = np.array(cc)
    pr = np.array(pr)
    dc_trans = dc.reshape(-1,1)
    cc_trans = cc.reshape(-1,1)
    pr_trans = pr.reshape(-1,1)

    ccpr = np.append(cc_trans, pr_trans, 1)

    sample_feature_arra = np.append(sample_feature_arr, dc_trans, 1)
    sample_feature_array = np.append(sample_feature_arra, ccpr, 1)

    print("Sample feature array dimensions")
    print(sample_feature_array.shape, flush=True)

    num_samples = int(len(sample_feature_array) / 2)
    target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)

    X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)
    get_basic_model_results(X_train, X_test, y_train, y_test)


def plot_feature_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)

    plt.savefig('feature_importance.png', bbox_inches='tight')
    plt.show()


def dump_random_forest_feature_importance(data_dir, news_source):
    include_micro = True
    include_macro = True

    include_structural = True
    include_temporal = True
    include_linguistic = True

    sample_feature_array = get_TPNF_dataset(data_dir, news_source, include_micro, include_macro, include_structural,
                                            include_temporal, include_linguistic, use_cache=True)

    sample_feature_array = sample_feature_array[:, :-1]
    feature_names, short_feature_names = get_dataset_feature_names(include_micro, include_macro, include_structural,
                                                                   include_temporal, include_linguistic)

    feature_names = feature_names[:-1]
    short_feature_names = short_feature_names[:-1]
    num_samples = int(len(sample_feature_array) / 2)
    target_labels = np.concatenate([np.ones(num_samples), np.zeros(num_samples)], axis=0)

    X_train, X_test, y_train, y_test = get_train_test_split(sample_feature_array, target_labels)

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=100, random_state=0)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    matplotlib.rcParams['figure.figsize'] = 5, 2

    # Plot the feature importances of the forest
    plt.figure()

    plt.bar(range(X_train.shape[1]), importances[indices],
            color="b", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), np.array(short_feature_names)[indices], rotation=75, fontsize=9.5)
    plt.xlim([-1, X_train.shape[1]])
    plt.savefig('{}_feature_importance.png'.format(news_source), bbox_inches='tight')

    plt.show()


def get_classificaton_results_tpnf_by_time(news_source: str):
    # Time Interval in hours for early-fake news detection
    time_intervals = [3, 6, 12, 24, 36, 48, 60, 72, 84, 96]

    for time_interval in time_intervals:
        print("=============Time Interval : {}  ==========".format(time_interval))
        start_time = time.time()
        get_classificaton_results_tpnf("data/features", news_source, time_interval)

        print("\n\n================Exectuion time - {} ==================================\n".format(
            time.time() - start_time))


if __name__ == "__main__":

    #dump_random_forest_feature_importance("data/features", "politifact")

    print("\n\n Working on Politifact Data \n")
    get_classificaton_results_tpnf("data/features", "politifact", time_interval=None, use_cache=False)

    print("\n\n Working on Gossipcop Data \n")
    get_classificaton_results_tpnf("data/features", "gossipcop", time_interval=None, use_cache=False)

    # Filter the graphs by time interval (for early fake news detection) and get the classification results
    # get_classificaton_results_tpnf_by_time("politifact")
    # get_classificaton_results_tpnf_by_time("gossipcop")
