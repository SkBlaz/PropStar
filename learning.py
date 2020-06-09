## misc methods for learning
from sklearn import preprocessing
from collections import defaultdict
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import subprocess
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score

def preprocess_and_split(X,num_fold=10, target_attribute = None):
    
    skf = StratifiedKFold(n_splits=num_fold)
    Y = X[target_attribute]
    if len(np.unique(Y)) > 40:
        tmp1 = [float(x) for x in Y]
        tmp = []
        for j in tmp1:
            if j > np.mean(tmp1):
                tmp.append(1)
            else:
                tmp.append(0)
        Y = tmp                 

    for train_index, test_index in skf.split(X, Y):
        yield train_index,test_index

def svm_learner(train_features,train_classes):
    clf = svm.SVC(kernel='rbf',C=10,gamma="scale")
    clf.fit(train_features,train_classes)
    return clf

def lr_learner (train_features,train_classes):
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_features,train_classes)
    return clf

class starspaceLearner:

    '''
    This is a simple wrapper for the starspace learner.
    '''
    
    def __init__(self,vb=False, binary="./starspace", tmp_folder ="tmp",epoch=5, dim=100, learning_rate = 0.01, neg_search_limit = 50, max_neg_samples = 10):
        self.binary = binary
        self.tmp = tmp_folder
        self.epoch = epoch
        self.dim = dim
        self.lr = learning_rate
        self.nsl = neg_search_limit
        self.nspb = max_neg_samples
        self.parameter_string = "-epoch {} -negSearchLimit {} -maxNegSamples {} -lr {} -dim {}".format(self.epoch, self.nsl, self.nspb, self.lr, self.dim)

        print(self.parameter_string)

    def data_to_text(self, train_data, label_tag=False):

        if not label_tag:
            rows, cols = np.nonzero(train_data)
            row_matrix = defaultdict(list)
            number_of_rows = train_data.shape[0]
            for enx, el in enumerate(rows):
                row_matrix[el].append(cols[enx])
                
            input_text = []
            for j in range(number_of_rows):
                if len(row_matrix[j]) == 0:
                    elements = ["0"]
                else:
                    elements = row_matrix[j]
                input_string = " ".join([str(x) for x in elements])
                input_text.append(input_string)
            return input_text
        else:
            labels = ["__label__"+str(x) for x in train_data]
#            labels = "__label__"+train_data.astype(str).values
            return labels

    def write_to_tmp(self,filedump,tag="train"):
        
        fx = open(self.tmp+"/"+tag+"_data.txt","w+")
        fx.write(filedump)
        fx.close()

    def call_starspace_binary(self,train=True, output_model = "tmp/storedModel"):

        if train:
            train_file = "./tmp/train_data.txt"
            to_execute = "./starspace train {} -verbose 0 -trainFile ".format(self.parameter_string)+train_file+" -model "+output_model
            os.system(to_execute)
        else:

            test_file = "./tmp/test_data.txt"
            to_execute = """
            ./query_predict {modelfile} 1 < {testfile}
            """.format(modelfile="tmp/storedModel",testfile="tmp/test_data.txt")
            output = subprocess.check_output(to_execute, shell=True)
            return output
            
    def fit(self,train_data,train_labels):
        train_text = self.data_to_text(train_data)
        train_label_text = self.data_to_text(train_labels,label_tag=True)
        total_data = []
        for enx, el in enumerate(train_text):
            total_list = el+" "+ train_label_text[enx]
            total_data.append(total_list)
        out_file = "\n".join(total_data)
        self.write_to_tmp(out_file)
        self.call_starspace_binary()

    def _cleanup(self):
#        os.remove("./tmp/train_data.txt")
#        os.remove("./tmp/test_data.txt")
#        os.remove("./tmp/storedModel.tsv")
#        os.remove("./tmp/storedModel")
        os.system("rm -rf ./tmp/*")
        time.sleep(3)
        
    def predict(self,test_data, return_scores = False, clean_tmp = False, return_int_predictions = True):

        train_text = self.data_to_text(test_data)
        total_data = []
        for enx, el in enumerate(train_text):
            total_list = el #+" "+ train_label_text[enx]
            total_data.append(total_list)
        out_file = "\n".join(total_data)

        self.write_to_tmp(out_file, tag="test")
        otpt = str(self.call_starspace_binary(train=False))
        els = []
        for el in otpt.split("\\n"):
            if "Enter" in el and "__label__" in el:
                if not return_scores:
                    el1 = el.split(" ")[-2].replace("__label__","")
                else:
                    el1 = float(el.split(" ")[-3].split("[")[1].split("]")[0])
                els.append(el1)
        if clean_tmp:
            self._cleanup()

        if return_int_predictions:
            return np.array([int(x) for x in els])
        else:
            return els
