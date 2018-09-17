import glob as gl
import nltk
import re
import numpy as npy
import matplotlib.pyplot as matplot
from nltk.corpus import stopwords
from scipy.io import savemat
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from scipy.sparse import dok_matrix
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')


class NaiveBayes:

    def __init__(self):
        self.data_matrix = None
        self.class_labels = None
        self.total_instances = 0
        self.pos_class_prob = 0
        self.neg_class_prob = 0
        self.vocabulary = None
        self.vocab_prob = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.train_fraction_accuracy_map = None
        self.pos_files = None
        self.all_files = None

    def remove_stop_words(self,review_file):

        with open(review_file,encoding='utf-8') as fl:
            review_text = fl.read().replace('\n','')
            review_text = re.sub('[^A-Za-z]+',' ',review_text)
            review_words = review_text.lower().split()
            stop_words = set(stopwords.words('english'))
            removed_words =[word for word in review_words if word not in stop_words and len(word) > 2]
            return " ".join(removed_words)

    def pre_process_data(self):

        word_dict = dict()
        data_matrix = dict()
        word_index = 0

        pos_files = gl.glob('pos\*.txt')
        neg_files = gl.glob('neg\*.txt')

        self.pos_files = pos_files

        #pos_files = pos_files[0:100] # 2000
        #neg_files = neg_files[0:100] # 2000

        pos_files.extend(neg_files)

        self.all_files = pos_files

        for fl in pos_files:
            review_text = self.remove_stop_words(fl)
            review_tokenized = wordpunct_tokenize(review_text)
            for word in review_tokenized:
                if word not in word_dict:
                    word_dict[word] = word_index
                    word_index = word_index + 1
                if (word, fl) not in data_matrix.keys():
                    data_matrix[(word, fl)] = 1
                else:
                    data_matrix[(word, fl)] = data_matrix[(word, fl)] + 1

        self.data_matrix = dok_matrix((len(word_dict.keys()), len(pos_files)))
        for word,fl in data_matrix.keys():
            word_index = word_dict[word]
            doc_index = pos_files.index(fl)
            self.data_matrix[word_index,doc_index] = data_matrix[(word,fl)]

        savemat('dm.mat', mdict={'arr': self.data_matrix})
        self.vocabulary = list(word_dict.keys())
        self.data_matrix = self.data_matrix.transpose()

    def train_model(self, train_ratio):
        labels = npy.zeros(len(self.all_files),dtype=npy.int) # 4000
        labels[0:len(self.pos_files)].fill(1) # 2000

        self.class_labels = labels

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.data_matrix, self.class_labels, train_size=train_ratio,random_state=58)

        self.x_train = self.x_train.transpose()
        self.x_test = self.x_test.transpose();

        class_label, class_count = npy.unique(self.y_train,return_counts=True)
        self.total_instances = self.x_train.shape[1]
        class_prob = dict(zip(class_label,class_count))

        # Class prior probability
        for label,count in class_prob.items():
            prob = count / self.total_instances
            if label == 1:
                self.pos_class_prob = prob
            else:
                self.neg_class_prob = prob

        # Posterior probability
        freq_count_all_class_pos = 0
        freq_count_all_class_neg = 0
        word_freq_map = {}
        self.vocab_prob = dict({})
        for word_index in range(0,len(self.vocabulary)):
            freq_word = [0,0]
            freq_sum = 0
            for document_index in range(0,self.x_train.shape[1]):
                class_index = int(self.y_train[document_index])
                freq_sum = freq_word[class_index] + int(self.x_train[word_index,document_index])
                freq_word[class_index] = freq_sum

                if class_index == 1:
                    freq_count_all_class_pos = freq_count_all_class_pos + freq_sum
                else:
                    freq_count_all_class_neg = freq_count_all_class_neg + freq_sum

            word_freq_map[self.vocabulary[word_index]] = freq_word
        word_freq_map = dict(word_freq_map)

        for key,list_class_prob in word_freq_map.items():
            self.vocab_prob[key] = [0,0]
            self.vocab_prob[key][0] = (list_class_prob[0] + 1)/(freq_count_all_class_neg + len(self.vocabulary))
            self.vocab_prob[key][1] = (list_class_prob[1] + 1)/(freq_count_all_class_pos + len(self.vocabulary))

    def classify(self):
        predict_class_labels = []

        for review in range(0,self.x_test.shape[1]):
            class_0_prob = self.pos_class_prob
            class_1_prob = self.neg_class_prob
            for word_index in range(0, len(self.vocabulary)):
                word = self.vocabulary[word_index]
                class_0_prob *= self.vocab_prob[word][0]
                class_1_prob *= self.vocab_prob[word][1]

            if class_1_prob > class_0_prob:
                predict_class_labels.append(1)
            else:
                predict_class_labels.append(0)

        self.y_predict = predict_class_labels

    def calculate_accuracy(self,train_fraction):
        success_count = 0

        for i in range(0,len(self.y_predict)):
            if self.y_predict[i] == self.y_test[i]:
                success_count = success_count + 1

        self.train_fraction_accuracy_map[train_fraction] = success_count / len(self.y_predict)


if __name__ == '__main__':

    naive_bayes = NaiveBayes()
    naive_bayes.pre_process_data()

    split_ratios = [0.1,0.3,0.5,0.7,0.9]

    average_accuracies = []
    run_accuracy = []
    for split_ratio in split_ratios:
         run_accuracy = []
         for i in range(0,5):
             naive_bayes.train_model(split_ratio)
             naive_bayes.classify()
             naive_bayes.train_fraction_accuracy_map = dict({})
             naive_bayes.calculate_accuracy(train_fraction=split_ratio)
             run_accuracy.append(naive_bayes.train_fraction_accuracy_map[split_ratio])

         average_accuracies.append(npy.mean(run_accuracy))

    matplot.plot(split_ratios[0:],average_accuracies)
    matplot.plot(split_ratios[0:],average_accuracies,'ro')
    matplot.xlabel('train data set - split ratio')
    matplot.ylabel('accuracy')
    matplot.savefig('accuracy_graph.png')
    matplot.show()

