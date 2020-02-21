import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
import pickle

st.title("Spam or not spam?")
st.write("This web app is aimed to detect spam and explore how it's been detected")

# getting dictionaries and prior probabilities from pretrained classifier
@st.cache(show_spinner=False)
def read_clf():
    with open('trained_classifier.pkl', 'rb') as f: # /Users/macbook/Downloads/
        pretrained_clf = pickle.load(f)
    return pretrained_clf

pretrained_clf = read_clf()
prior_spam = pretrained_clf[0]
prior_ham = pretrained_clf[1]
spam_dict = pretrained_clf[2]
ham_dict = pretrained_clf[3]
unique_words = pretrained_clf[4]

# conditional probability for single word
def conditional_word(word, label, k=1):
    if label == 'spam':
        if word.lower() in spam_dict:
            cond_word_spam = np.log(
                (k + spam_dict[word.lower()]) / (len(unique_words) + sum(spam_dict.values())))
        else:
            cond_word_spam = np.log((k) / (len(unique_words) + sum(spam_dict.values())))

        cond_word_spam = cond_word_spam
        return cond_word_spam
    else:
        if word.lower() in ham_dict:
            cond_word_ham = np.log((k + ham_dict[word.lower()]) / (
                    len(unique_words) + sum(ham_dict.values())))
        else:
            cond_word_ham = np.log((k) / (len(unique_words) + sum(ham_dict.values())))
        cond_word_ham = cond_word_ham
        return cond_word_ham

# conditional probability for text
def conditional(text, label):
    if label == 'spam':
        cond_spam = 0
        for word in text.split(' '):
            word = re.sub(r'[^\w\s]', '', word)
            if len(word) > 2 and len(re.findall(r'\d+', word)) == 0:
                cond_spam += conditional_word(word.lower(), label)
        return cond_spam
    else:
        cond_ham = 0
        for word in text.split(' '):
            word = re.sub(r'[^\w\s]', '', word)
            if len(word) > 2 and len(re.findall(r'\d+', word)) == 0:
                cond_ham += conditional_word(word.lower(), label)
        return cond_ham

# getting probability of email being spam
def classify(text):
        ham = np.log(prior_ham) + conditional(text, 'ham')
        spam = np.log(prior_spam) + conditional(text, 'spam')
        prob_spam = 1 / (1 + np.exp(ham - spam))
        return prob_spam

text_input = st.text_area("Place the text here to see if it is spam")
if text_input:
    probability = classify(text_input)
    st.subheader('The probability that this message is spam is {:.2f}%'.format(probability*100))
    st.write('Although we never know the probability of an email being spam and usually '
             'see it either in the inbox or junk folder, some algorithms detect spam based on the probability. '
             'How such an algorithm works and how the math behind it looks like is shown below')

# getting classified data
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_data():
    processed_data = pd.read_csv('https://raw.githubusercontent.com/ipushin/Spam-Detector/master/test_spam.csv')
    return processed_data

# assigning classes on the basis of the probability
@st.cache(allow_output_mutation=True, show_spinner=False)
def adjusted_classes(probs, t):
    return [1 if prob > t else 0 for prob in probs]

# getting conf matrix values for visualisation
@st.cache(allow_output_mutation=True, show_spinner=False)
def positive_negative(row):
    if row['adj_class'] == row['label'] and row['adj_class'] == 1:
        return 'TP'
    elif row['adj_class'] == row['label'] and row['adj_class'] == 0:
        return 'TN'
    elif row['adj_class'] != row['label'] and row['adj_class'] == 0:
        return 'FN'
    else:
        return 'FP'

def plot_auc():
    fig = make_subplots(rows=1, 
                        cols=2, 
                        horizontal_spacing=0.2,
                        subplot_titles=("Area Under Curve", "Precision and Recall <br> with threshold"))

    fig.add_trace(go.Scatter(x=fpr, 
                             y=tpr, 
                             mode='lines+markers', 
                             name='AUC',
                             fill='tozeroy',
                             fillcolor='rgba(110, 240, 120, 0.1)',
                             line=dict(color='green', width=3)), 1, 1)

    color = ['red', 'orange']
    names = ['Precision', 'Recall']
    for i, j in enumerate([precision, recall]):
        fig = fig.add_trace(go.Scatter(x=thresholds,
                                       y=j[:-1], 
                                       mode='lines', 
                                       line=dict(color=color[i], width=3),
                                       name=names[i]), 1, 2)

    fig.add_trace(go.Scatter(x=[0,1],
                             y=[0,1], 
                             name='Random <br> classifier', 
                             mode="lines", 
                             line=dict(color='grey', width=2, dash='dash')), 1, 1)

    axes_names = ['False Positive Rate', 'True Positive Rate', 'Decision Threshold', 'Score']
    for j,i in enumerate(['xaxis', 'yaxis', 'xaxis2', 'yaxis2']):
        fig.layout[i]['showgrid'] = True
        fig.layout[i]['gridwidth'] = 0.1
        fig.layout[i]['showgrid'] = True
        fig.layout[i]['gridcolor'] = '#EEEEEE'
        fig.layout[i]['showline'] = True
        fig.layout[i]['linewidth'] = 1
        fig.layout[i]['mirror'] = True
        fig.layout[i]['ticks'] = 'outside'
        fig.layout[i]['nticks'] = 8
        fig.layout[i]['range'] = [-0.03, 1.03]
        fig.layout[i]['linecolor'] = 'black'
        fig.layout[i]['title'] = axes_names[j]
        fig.layout[i]['title_standoff'] = 8
        fig.layout[i]['title_font'] = {"size": 11}

    for i in range(0, 2):
        fig.layout['annotations'][i]['font']['size'] = 14
        fig.layout['annotations'][i]['y'] = 1.05

    fig.update_layout(width=800, 
                      height=300, 
                      margin = dict(l=0, r=0, t=15, b=5),
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig)

def plot_treshold():
    fig = make_subplots(rows=1,
                        cols=3,
                        horizontal_spacing=0.1,
                        column_widths=[0.45, 0.025, 0.45],
                        specs=[[{'type': 'histogram'}, {'type': 'bar'}, {'type': 'scatter'}]])

    colors = ['orange', 'red', '#00A0FC', 'blue']
    plot_bar = pd.DataFrame(processed_data['pos_neg'].value_counts())

    # correct results and errors histogram
    for j, i in enumerate(processed_data['pos_neg'].unique()):
        fig.add_trace(go.Histogram(x=processed_data[processed_data['pos_neg'] == i]['prob_spam'],
                                    xbins_size=0.1,
                                    marker_color=colors[j],
                                    name=i,
                                    hoverinfo="y",
                                    showlegend=False
                                    ), 1, 1)

        # number of true and false values
        fig.add_trace(go.Bar(x=[''],
                             name=i,
                             y=[plot_bar['pos_neg'][i]],
                             marker=dict(color=colors[j])), 1, 2)

    # threshold position
    fig.add_trace(go.Scatter(x=[t, t], y=[-5, 500],
                             mode="lines+text",
                             text="Threshold",
                             textposition="top right",
                             showlegend=False,
                             line=dict(color='grey', width=2, dash='dash')), 1, 1)

    # precision, recall values at the given threshold
    def dot_coord(t):
        close_default_clf = np.argmin(np.abs(thresholds - t))
        if 0 <= t < 1:
            return recall[close_default_clf], precision[close_default_clf]
        else:
            return 0, 0

    # precision and recall as a function of the threshold
    fig.add_trace(go.Scatter(x=recall,
                             y=precision,
                             mode='lines',
                             line=dict(color='brown', width=3),
                             fill='tozeroy',
                             fillcolor='rgba(230, 160, 130, 0.1)',
                             name='Recall/Precision'), 1, 3)

    # precision, recall values at the given threshold
    fig.add_trace(go.Scatter(x=[None, dot_coord(t)[0]],
                             y=[None, dot_coord(t)[1]],
                             mode='markers',
                             name='Value at the <br> given threshold',
                             marker=dict(size=10, color='black')), 1, 3)

    # customizing graphs
    axes_names = ['Probability', 'Frequency', 'Recall', 'Precision', 'TP, TN, FN, FP, <br> quantity', '']
    for j, i in enumerate(['xaxis', 'yaxis', 'xaxis3', 'yaxis3', 'xaxis2', 'yaxis2']):
        if i not in ['xaxis2', 'yaxis2']:
            fig.layout[i]['showgrid'] = True
            fig.layout[i]['gridwidth'] = 0.05
            fig.layout[i]['showgrid'] = True
            fig.layout[i]['gridcolor'] = '#EEEEEE'
            fig.layout[i]['showline'] = True
            fig.layout[i]['linewidth'] = 0.2
            fig.layout[i]['mirror'] = True
            fig.layout[i]['ticks'] = 'outside'
            fig.layout[i]['linecolor'] = 'black'
            if 'xaxis' in i:
                fig.layout[i]['range'] = [0.0, 1.0]
        fig.layout[i]['nticks'] = 10
        fig.layout[i]['title'] = axes_names[j]
        fig.layout[i]['title_standoff'] = 4
        fig.layout[i]['title_font'] = {"size": 11}

    fig.update_layout(barmode='stack',
                      width=850,
                      margin=dict(l=0, r=0, t=15, b=5),
                      height=300,
                      plot_bgcolor='rgba(0,0,0,0)')

    fig.update_layout(yaxis_type="log")
    st.plotly_chart(fig)

@st.cache(allow_output_mutation=True, show_spinner=False)
def make_clickable(url, text):
    return f'<a target="_blank" href="{url}">{text}</a>'

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_metrcis(col='pred_class'):
    fpr, tpr, _ = metrics.roc_curve(processed_data['label'], processed_data['prob_spam'])
    precision, recall, thresholds = metrics.precision_recall_curve(processed_data['label'], processed_data['prob_spam'])
    roc_auc_score = metrics.roc_auc_score(processed_data['label'], processed_data['prob_spam'])
    precision_score = metrics.precision_score(processed_data['label'], processed_data[col])
    recall_score = metrics.recall_score(processed_data['label'], processed_data[col])

    conf_matrix = pd.DataFrame(metrics.confusion_matrix(processed_data['label'], processed_data[col]))
    conf_matrix.iloc[0, 0] = str(conf_matrix.iloc[0, 0]) + " (TN)"
    conf_matrix.iloc[1, 1] = str(conf_matrix.iloc[1, 1]) + " (TP)"
    conf_matrix.iloc[0, 1] = str(conf_matrix.iloc[0, 1]) + " (FP)"
    conf_matrix.iloc[1, 0] = str(conf_matrix.iloc[1, 0]) + " (FN)"
    conf_matrix.columns = ['pred_ham', 'pred_spam']
    conf_matrix.index = ['true_ham', 'true_spam']

    if col == 'adj_class':
        roc_auc_score = metrics.roc_auc_score(processed_data['label'], processed_data['adj_class'])
    return fpr, tpr, precision, recall, thresholds, roc_auc_score, precision_score, recall_score, conf_matrix

def color_table(val):
    if val == conf_matrix.iloc[0, 0]:
        color = 'orange'
    elif val == conf_matrix.iloc[1, 1]:
        color = 'red'
    elif val == conf_matrix.iloc[0, 1]:
        color = 'blue'
    elif val == conf_matrix.iloc[1, 0]:
        color = '#00A0FC'
    return 'color: %s' % color

algo = st.checkbox('Algorithm')
if algo:
    processed_data = get_data()

    st.write("To sort spam from non-spam emails we use an algorithm which calculates probability of the email being spam "
             "and decide what class ('spam' or 'Ham') the email belongs to. ")

    st.write("When classifying emails the algorithm can correctly detect true spam as spam which is a *True Positive* result or "
             "identify non-spam as non-spam (*True Negative*). When it fails it detects spam as non-spam "
             "(*False Negative* error) or labels a true non-spam email as spam (*False Positive*).")

    # plotting conf matrix of predicted classes
    conf_matrix = get_metrcis('pred_class')[8]
    st.write("Here is how the algorithm's correct results and errors are distributed")
    styled_table = conf_matrix.style.applymap(color_table)
    st.dataframe(styled_table)

    st.write('The algorithm assigns a class with a higher probability to the email. Since the sum of probabilities '
             'is 1, the default separating threshold is 0.5. But if we consider spam emails as a serious disease, '
             "the threshold could be as low as 0.1. We'll get the low number of spam emails identified as non-spam (FN) and the high volume "
             "of incorrectly labeled non-spam emails (FP).")

    t = st.slider("Try to change the threshold and see what happens", 0.0, 1.0001)
    processed_data['adj_class'] = adjusted_classes(processed_data['prob_spam'], t)
    processed_data['pos_neg'] = processed_data.apply(positive_negative, axis=1)

    precision = get_metrcis('adj_class')[2]
    recall = get_metrcis('adj_class')[3]
    thresholds = get_metrcis('adj_class')[4]
    roc_auc_score = get_metrcis('adj_class')[5]
    precision_score = get_metrcis('adj_class')[6]
    recall_score = get_metrcis('adj_class')[7]

    plot_treshold()
    st.markdown('**AUC score**: {:.2f} | **Precision score**: {:.2f} | **Recall score**: {:.2f}'.format(roc_auc_score,
                                                                                                        precision_score,
                                                                                                       recall_score))
    # plotting conf matrix of adjusted classes for specific threshold
    onf_matrix = get_metrcis('adj_class')[8]
    styled_table = conf_matrix.style.applymap(color_table)
    st.dataframe(styled_table)

    st.write("As the performance metrics we use the following combinations of the correct results and errors")

    st.markdown(r'''$\textcolor{orange}{TPR (recall)} = \dfrac{\textcolor{red}{TP}}{\textcolor{red}{TP}+\textcolor{#00A0FC}{FN}}$ |
                    $\textcolor{blue}{FPR} = \dfrac{\textcolor{blue}{FP}}{\textcolor{blue}{FP}+\textcolor{orange}{TN}}$ |
                    $\textcolor{red}{PPV (precision)} = \dfrac{\textcolor{red}{TP}}{\textcolor{red}{TP}+\textcolor{blue}{FP}}$  ''')

    st.markdown('**Recall or Sensitivity** answers this question: “Of all the messages that are truly spam, how many did we label correctly?”. '
                '**Precision** metric means “Of all spam messages that labeled as spam, how many are actually spam?”. '
                'The **false positive rate** (**FPR**) is the number of emails which are not spam but are identified as spam, ' 
                'divided by the total number of not spam messages.')

    st.write('The main metric to evaluate the performance of the classifier is the **AUC ROC** . What stands for Area Under the Receiver '
             'Operating Characteristic Curve. The ROC curve plots the performance of a binary classifier under various threshold settings '
             'and measured by TPR and FPR. AUC represents degree or measure of classes separability. The higher the AUC, the better the model '
             'is at predicting Spam as Spam and Ham as Ham.')

    fpr = get_metrcis()[0]
    tpr = get_metrcis()[1]
    precision = get_metrcis()[2]
    recall = get_metrcis()[3]
    thresholds = get_metrcis()[4]
    roc_auc_score = get_metrcis('pred_class')[5]
    precision_score = get_metrcis('pred_class')[6]
    recall_score = get_metrcis('pred_class')[7]
    plot_auc()
    st.markdown('**AUC**: {:.2f} | **Precision**: {:.2f} | **Recall**: {:.2f}'.format(roc_auc_score, precision_score, recall_score))
    st.markdown('**Interpretation of metrics**')
    st.markdown(r'''
    + AUC close to 1 means the classifier is pretty good at sorting spam from non-spam
    + If we set the threshold to 0 or 1 we get AUC=0.5 what means the classifier is as good as random choice 
    + The balance point between Precision and Recall is when the threshold is as low as 0.1 
    + True values are stored close to 0 and 1 which is good since classes are correctly separated
    ''')
    
    link = make_clickable('https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc', 'more')
    st.write("In the context of spam classification AUC is good enough. But there are much {} "
             "metrics to evaluate algorithm's performance.".format(link), unsafe_allow_html=True)
    st.write("Check the Theory section to see what's going on under the hood of the algorithm in details.")

theory = st.checkbox('Theory')
if theory:
  st.write("**Bayes' Theory**")
  st.write("The main idea for our spam classification algorithm is to apply Bayes's theorem to sort class'spam' from 'Ham' (not spam). "
           "The algorithm itself belongs to the group of the supervised machine learning tasks for pattern classification when a "
           "training model is based on labeled training data. The whole mechanism of training and classification is based on the following rule "
           "formulated by Thomas Bayes:")
  st.markdown(r'''$\mathsf{PosteriorProbability} = \dfrac{\mathsf{\textcolor{red}{PriorProbability}}  \times  
                   \mathsf{\textcolor{blue}{ConditionalProbability}}}{\mathsf{\textcolor{orange}{Evidence}}}$''')

  st.markdown("A posterior probability can be read as “What is the probability that a particular text composed of words($x_i$) belongs to class $S$ ?” "
              "and can be formulated as")
  st.markdown(r'''$P(S\mid x_i) = \dfrac{\textcolor{red}{P(S)} \times \textcolor{blue}{P(x_i\mid S)}}{\textcolor{orange}{P(x_i)}}$''')

  st.write('**Major assumption**')
  st.markdown('A conditional probability is calculated under the major assumption that words in text are conditionally independent of each other. '
              'Independence means that the probability of one observation does not affect the probability of another observation. That is why the '
              'Bayesian classifier also called naive algorithm because it tranform a coherent text into '
              '“a bag of words”. Following this assumption we can calculate conditional probabilities by using a *chain rule* and multiplying '
              'probabilities of individual words.')

  st.write('**Prior Probability**')
  st.write('A prior probability can be interpreted as the prior belief or *a priori* knowledge or “the general probability of encountering a '
           'particular class.” A prior probability can be estimated as')
  st.markdown(r'''$\textcolor{red}{P(S)} = \dfrac{N_s}{N_c}$''')
  st.markdown(r'''where
  + $N_s$: count of spam messages in training data
  + $N_c$: count of all messages in training data
  ''')

  st.write('**Conditional Probability**')
  st.markdown(r'''A conditional probability $\textcolor{blue}{P(x_i\mid S)}$ can be formulated as “How likely is it to observe this 
                  particular feature $x$ given that it belongs to class $S$ ?”''')
  st.markdown(r'The classical approach to estimate a conditional probability is to calculate the following frequency')
  st.markdown(r'''$\textcolor{blue}{P(x_i\mid S)} = \dfrac{N_{x_i,S}}{N_S}$''')
  st.markdown(r'''where
  + $N_{x_i,S}$ is the number of times feature $x_i$ (word) appears in samples from class $S$
  + $N_S$ is the total count of all features in class $S$
  ''')

  st.write("But it doesn't help us to deal with the words out of the training dataset. That is why an alternative method to "
           "estimate a conditional probability is used")
  st.markdown(r'''$\textcolor{blue}{P(x_i\mid S)} = \dfrac{\sum tf(x_i, d \in S) + \mathit{\alpha}}{\sum N_{d \in S} + \mathit{\alpha} \cdot V}$''')
  st.markdown(r'''where
  + $x_i$: A word from the feature vector $x$ of a particular sample.
  + $\sum tf(x_i, d \in S)$: The sum of raw term frequencies of word $x_i$ from all documents in the training sample that belong to class $S$
  + $\sum N_{d \in S}$: The sum of all term frequencies in the training dataset for class $S$
  + $\mathit{\alpha}$: An additive smoothing parameter ($\mathit{\alpha}=1$ for Laplace smoothing)
  + $V$: The size of the vocabulary (number of different words in the training set)
  ''')

  st.markdown(r'''The posterior probability of the given text can be calculated as a product of the probabilities of the individual words
   under the following *chain rule*:''')
  st.markdown(r'''$\textcolor{blue}{P(x_i\mid S)} = P(x_1\mid S)\times P(x_2\mid S)\times ... \times P(x_d\mid S) = \displaystyle\prod_{i=1}^d P(x_i\mid S)$''')

  st.write('**Evidence**')
  st.markdown(r'''The evidence $\textcolor{orange}{P(x)}$ can be understood as the probability of encountering a particular pattern $x$ independent from the class label.
  Although the evidence term is required to accurately calculate the posterior probabilities, it can be removed since it is merely a scaling factor when
  we compare posterior probabilities of two classes''')
  st.markdown(r'''$\dfrac{P(x_i\mid S)\times P(S)}{\textcolor{orange}{P(x_i)}} \gt \dfrac{P(x_i\mid H)\times P(H)}{\textcolor{orange}{P(x_i)}} \implies P(x_i\mid S)\times P(S) \gt P(x_i\mid H)\times P(H)$ ''')

  st.write('**Floating point underflow**')
  st.markdown('Multiplying a set of small probabilities together will probably result in '
              '*floating point underflow* where the product will become too small to represent and will be replaced by 0. Instead of estimating ')
  st.markdown(r'''$P(S)\displaystyle\prod_{i=1}^nP(x_i\mid S)$, we consider computing the logarithm of''')
  st.markdown(r'''$\log(P(S)\displaystyle\prod_{i=1}^nP(x_i\mid S))$ which can be written equivalently as''')
  st.markdown(r'''$\log(P(S))+\displaystyle\sum_{i=1}^n \log(P(x_i\mid S))$''')

  st.write('**Decision rule**')
  st.markdown('The final decision on what class an email belongs to is based on the following rule')
  st.markdown(r'''$\log(\textcolor{red}{P(S)})+\displaystyle\sum_{i=1}^n \log(\textcolor{blue}{P(x_i\mid S)}) \gt
                   \log(\textcolor{red}{P(H)})+\displaystyle\sum_{i=1}^n \log(\textcolor{blue}{P(x_i\mid H)})$''')
  st.markdown('The class which has a greater sum of logarithms is assigned to a particular email')

  st.markdown("**Maximum a posteriori estimation (MAP)**")
  st.markdown("The probability shown at the beginning is not 100% correct, because of the slight difference between the concepts of "
              "probability and likelihood. In order to classify the given text the concept of maximum a posteriori estimation was employed, "
              "which in turn closely related to the maximum likelihood estimation. By calculating MAP we maximize a posterior probability. "
              "What means we are taking into account only prior and conditional probability without calculating the evidence.")

  st.markdown("In case we'd like to transform log values into decimal or percent form the following formulas can be used:")
  st.markdown(r'''$S_{map} = \dfrac{1}{1+e^{q_h - q_s}}$''')
  st.markdown(r'''$H_{map} = \dfrac{1}{1+e^{q_s - q_h}}$''')
  st.markdown(r'''where $q_s$ and $q_h$ is a sum of logarithms of a prior and conditional probability for $S$ and $H$ class.''')
  st.write("Having MAP calculated in decimal form we can set up a custom threshold from 0 to 1 to sort out classes as shown in the 'Algorithm' section above. ")

  link_one = make_clickable('https://sebastianraschka.com/Articles/2014_naive_bayes_1.html', 'post')
  st.write('*The theory section is mostly based on this {} made by Sebastian Rashka.*'.format(link_one), unsafe_allow_html=True)

  st.write("Check out the source code below to explore how the theory is implemented.")

code = st.checkbox('Source code')
if code:
  with st.echo():

      # this class contains functions implementing Bayes' theorem, calculating
      # probabilities to decide if the given text is spam

      class SpamClassifier():

          def fit(self, train):

              # words frequency
              self.spam_dict = {}
              self.ham_dict = {}
              train['email'] = train['email'].astype('str')
              for i in range(0, train.shape[0]):
                  if train['label'][i] == 1:
                      for j in train['email'][i].split(' '):
                          # tokenizing and filtering words
                          j = re.sub(r'[^\w\s]|_', '', j)
                          if len(j) > 2 and len(re.findall(r'\d+', j)) == 0:
                              self.spam_dict[j.lower()] = self.spam_dict.setdefault(j.lower(), 0) + 1

                  else:
                      for j in train['email'][i].split(' '):
                          j = re.sub(r'[^\w\s]|_', '', j)
                          if len(j) > 2 and len(re.findall(r'\d+', j)) == 0:
                              self.ham_dict[j.lower()] = self.ham_dict.setdefault(j.lower(), 0) + 1

              # prior probabilities
              ham_count = train[train['label'] == 0].shape[0]
              spam_count = train[train['label'] == 1].shape[0]
              self.prior_spam = spam_count / train.shape[0]
              self.prior_ham = ham_count / train.shape[0]

              # unique_words
              self.unique_words = {}
              for i in range(0, train.shape[0]):
                  for j in train['email'][i].split(' '):
                      j = re.sub(r'[^\w\s]|_', '', j)
                      if len(j) > 2 and len(re.findall(r'\d+', j)) == 0:
                          self.unique_words[j.lower()] = self.unique_words.setdefault(j.lower(), 0) + 1

          # conditional probability for individual word
          def conditional_word(self, word, label, k=1):
              if label == 'spam':
                  if word in self.spam_dict:
                      cond_word_spam = np.log(
                          (k + self.spam_dict[word]) / (len(self.unique_words) + sum(self.spam_dict.values())))
                  else:
                      cond_word_spam = np.log((k) / (len(self.unique_words) + sum(self.spam_dict.values())))

                  self.cond_word_spam = cond_word_spam
                  return self.cond_word_spam

              else:
                  if word in self.ham_dict:
                      cond_word_ham = np.log((k + self.ham_dict[word]) / (
                              len(self.unique_words) + sum(self.ham_dict.values())))
                  else:
                      cond_word_ham = np.log((k) / (len(self.unique_words) + sum(self.ham_dict.values())))
                  self.cond_word_ham = cond_word_ham
                  return self.cond_word_ham

          # conditional probability for text, chain rule implementation
          def conditional(self, text, label):
              if label == 'spam':
                  self.cond_spam = 0
                  for word in text.split(' '):
                      word = re.sub(r'[^\w\s]', '', word)
                      if len(word) > 2 and len(re.findall(r'\d+', word)) == 0:
                          self.cond_spam += SpamClassifier.conditional_word(word.lower(), label)
                  return self.cond_spam
              else:
                  self.cond_ham = 0
                  for word in text.split(' '):
                      word = re.sub(r'[^\w\s]', '', word)
                      if len(word) > 2 and len(re.findall(r'\d+', word)) == 0:
                          self.cond_ham += SpamClassifier.conditional_word(word.lower(), label)
                  return self.cond_ham

          # classifying text and getting spam probability
          def classify(self, text):
              ham = np.log(self.prior_ham) + SpamClassifier.conditional(text, 'ham')
              spam = np.log(self.prior_spam) + SpamClassifier.conditional(text, 'spam')
              prob_spam = 1 / (1 + np.exp(ham - spam))
              return prob_spam
