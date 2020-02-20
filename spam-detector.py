import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle

st.title("Spam or not spam?")
st.write("This web app is aimed to detect spam and explore how it's been detected")

@st.cache(allow_output_mutation=True, show_spinner=False)
def read_data():
 return pd.read_csv('https://raw.githubusercontent.com/ipushin/Spam-Detector/master/spam_or_not_spam.csv')
df = read_data()

#return prior_spam, prior_ham, spam_dict, ham_dict, unique_words
with open('trained_classifier.pkl', 'rb') as f:
    classifier_data = pickle.load(f)

prior_spam, prior_ham, spam_dict, ham_dict, unique_words = classifier_data[0], classifier_data[1], classifier_data[2], \
                                                           classifier_data[3], classifier_data[4]

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

def classify(text):
        ham = np.log(prior_ham) + conditional(text, 'ham')
        spam = np.log(prior_spam) + conditional(text, 'spam')
        prob_spam = 1 / (1 + np.exp(ham - spam))
        return prob_spam

text_input = st.text_area("Place the text here to see if it is spam")
if text_input:
    probability = classify(text_input)
    st.subheader('The probability this message is spam is {:.2f}%'.format(probability*100))
    st.write('Although we never know the probability of the email being spam and usually '
             'see it either in the inbox or junkbox, some algorithms detect spam based on the probability. '
             'How such an algorithm works and how the math behind it looks like is shown below')

@st.cache(allow_output_mutation=True, show_spinner=False)
def processed_data():
    processed_data = pd.read_csv('https://raw.githubusercontent.com/ipushin/Spam-Detector/master/test_spam.csv')
    return processed_data

@st.cache(allow_output_mutation=True, show_spinner=False)
def adjusted_classes(y_scores, t):
    return [1 if y >= t else 0 for y in y_scores]

@st.cache(allow_output_mutation=True, show_spinner=False)
def posivite_negative(row):
    if row['adj_class'] == row['label'] and row['adj_class'] == 1:
        return 'TP'
    elif row['adj_class'] == row['label'] and row['adj_class'] == 0:
        return 'TN'
    elif row['adj_class'] != row['label'] and row['adj_class'] == 0:
        return 'FN'
    else:
        return 'FP'

def plot_one():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing = 0.2,
                    subplot_titles=("Area Under Curve", "Precision and Recall <br> with threshold"))

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines+markers', name='AUC',
                                fill='tozeroy',fillcolor='rgba(110, 240, 120, 0.1)',
                                line = dict(color='green', width=3)),1,1)

    color = ['red','orange']
    names = ['Precision', 'Recall']
    for i,j in enumerate([precision, recall]):
        fig = fig.add_trace(go.Scatter(x=thresholds,
                                   y=j[:-1], mode='lines', line = dict(color=color[i], width=3),
                                   name=names[i]),1,2)

    fig.add_trace(go.Scatter(x=[0,1],y=[0,1], name = 'Random <br> classifier', mode="lines", line = dict(color='grey', width=2, dash='dash')),1,1)

    fig.update_layout(width=850, height=450,plot_bgcolor='rgba(0,0,0,0)')
    axes_names = ['False Positive Rate', 'True Positive Rate', 'Decision Threshold', 'Score' ]
    for j,i in enumerate(['xaxis','yaxis','xaxis2','yaxis2']):
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
    fig.update_layout(width=800, height=300,margin = dict(l=0, r=0, t=15, b=5))
    st.plotly_chart(fig)

def plot_two():
    fig2 = make_subplots(rows=1, cols=3, horizontal_spacing=0.1, column_widths=[0.45, 0.025, 0.45],
                     specs=[[{'type': 'histogram'}, {'type': 'bar'}, {'type': 'scatter'}]],
                     subplot_titles=("TP/TN Tradeoff",
                                     "",
                                     'Precision and Recall Scores as <br> a function of the decision threshold'))

    colors = ['orange', 'red', '#00A0FC', 'blue']
    plot_bar = pd.DataFrame(processed_data['pos_neg'].value_counts())

    for j, i in enumerate(processed_data['pos_neg'].unique()):
        fig2.add_trace(go.Histogram(x=processed_data[processed_data['pos_neg'] == i]['prob_spam'],
                                    xbins_size=0.08,
                                    marker_color=colors[j],
                                    name=i,
                                    opacity=0.8,
                                    showlegend=False
                                    ), 1, 1)

        fig2.add_trace(go.Bar(x=[''], name=i, y=[plot_bar['pos_neg'][i]], marker=dict(color=colors[j])), 1, 2)

    fig2.add_trace(go.Scatter(x=[t, t], y=[-5, 500], mode="lines+text",
                          text="Threshold", textposition="top right",
                          showlegend=False,
                          line=dict(color='grey', width=2, dash='dash')), 1, 1)

    close_default_clf = np.argmin(np.abs(thresholds - t))
    fig2.add_trace(go.Scatter(x=recall,
                          y=precision,
                          mode='lines',
                          line=dict(color='brown', width=3),
                          fill='tozeroy', fillcolor='rgba(230, 160, 130, 0.1)',
                          name='Recall/Precision'), 1, 3)

    fig2.add_trace(go.Scatter(x=[None, recall[close_default_clf]],
                          y=[None, precision[close_default_clf]],
                          mode='markers',
                          name='Value at the <br> given Threshold',
                          marker=dict(size=10, color='yellow')), 1, 3)

    axes_names = ['Probability', 'Frequency', 'Recall', 'Precision', 'TP, TN, FN, FP, <br> quantity', '']
    for j, i in enumerate(['xaxis', 'yaxis', 'xaxis3', 'yaxis3', 'xaxis2', 'yaxis2']):
        if i not in ['xaxis2', 'yaxis2']:
            fig2.layout[i]['showgrid'] = True
            fig2.layout[i]['gridwidth'] = 0.05
            fig2.layout[i]['showgrid'] = True
            fig2.layout[i]['gridcolor'] = '#EEEEEE'
            fig2.layout[i]['showline'] = True
            fig2.layout[i]['linewidth'] = 1
            fig2.layout[i]['mirror'] = True
            fig2.layout[i]['ticks'] = 'outside'
            fig2.layout[i]['linecolor'] = 'black'
            if 'xaxis' in i:
                fig2.layout[i]['range'] = [-0.02, 1.02]
        fig2.layout[i]['nticks'] = 10
        fig2.layout[i]['title'] = axes_names[j]
        fig2.layout[i]['title_standoff'] = 4
        fig2.layout[i]['title_font'] = {"size": 11}
    for i in range(0, 2):
        fig2.layout['annotations'][i]['font']['size'] = 14
        fig2.layout['annotations'][i]['y'] = 1.05

    fig2.update_layout(barmode='stack',
                       width=850,
                       margin=dict(l=0, r=0, t=15, b=5),
                       height=300,
                       plot_bgcolor='rgba(0,0,0,0)')

    fig2.update_layout(yaxis_type="log")
    st.plotly_chart(fig2)

@st.cache(allow_output_mutation=True, show_spinner=False)
def make_clickable(url, text):
    return f'<a target="_blank" href="{url}">{text}</a>'

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_metrcis(col='pred_class'):
    fpr, tpr, thresholds = metrics.roc_curve(processed_data['label'], processed_data['prob_spam'])
    precision, recall, thresholds = metrics.precision_recall_curve(processed_data['label'], processed_data['prob_spam'])

    precision_score = metrics.precision_score(processed_data['label'], processed_data[col])
    recall_score = metrics.recall_score(processed_data['label'], processed_data[col])
    accuracy = metrics.accuracy_score(processed_data['label'], processed_data[col])
    conf_matrix = pd.DataFrame(metrics.confusion_matrix(processed_data['label'], processed_data[col]))
    conf_matrix.iloc[0, 0] = str(conf_matrix.iloc[0, 0]) + " (TN)"
    conf_matrix.iloc[1, 1] = str(conf_matrix.iloc[1, 1]) + " (TP)"
    conf_matrix.iloc[0, 1] = str(conf_matrix.iloc[0, 1]) + " (FP)"
    conf_matrix.iloc[1, 0] = str(conf_matrix.iloc[1, 0]) + " (FN)"
    conf_matrix.columns = ['pred_ham', 'pred_spam']
    conf_matrix.index = ['true_ham', 'true_spam']

    if col == 'pred_class':
        roc_auc_score = metrics.roc_auc_score(processed_data['label'], processed_data['prob_spam'])
    else:
        roc_auc_score = metrics.roc_auc_score(processed_data['label'], processed_data[col])

    return fpr, tpr, precision, recall, thresholds, roc_auc_score, precision_score, recall_score, accuracy, conf_matrix

def color_table(val):
    if val == conf_matrix.iloc[0, 0]:
        color = 'orange'
    elif val == conf_matrix.iloc[1, 1]:
        color = 'red'
    elif val == conf_matrix.iloc[0, 1]:
        color = 'blue'
    elif val == conf_matrix.iloc[1, 0]:
        color = '#00A0FC'
    else:
        color = 'green'
    return 'color: %s' % color

algo = st.checkbox('Algorithm')
if algo:
    processed_data = processed_data()
    st.write("To sort spam from non-spam emails we use an algorithm which calculates probability of the email being spam "
             "and decide what class ('Spam' or 'Ham') the email belongs to. ")
    st.write("When classifying emails the algorithm can correctly detect true spam as spam which is a *True Positive* result or "
             "identify non-spam as non-spam (*True Negative*). When it fails it detects spam as non-spam "
             "(*False Negative* error) or labels a true non-spam email as spam (*False Positive*).")

    conf_matrix = get_metrcis('pred_class')[9]
    st.write("Here is how the algorithms' correct results and errors are distributed")
    styled_table = conf_matrix.style.applymap(color_table)
    st.dataframe(styled_table)

    #st.markdown('where **TN** is True Negative | **TP**: True Positive | **FP**: False Positive | **FN**: False Negative')

    st.write('The algorithm assigns a class with a higher probability to the email. Since the sum of probabilities '
             'is 1, the default separating threshold is 0.5. But if we consider spam emails as a serious disease, '
             "the threshold could be as low as 0.1. We'll get the low number of spam emails identified as non-spam (FN) and the high volume "
             "of incorrectly labeled non-spam emails (FP).")

    t = st.slider("Try to change the threshold and see what happens", 0.0, 1.0)
    processed_data['adj_class'] = adjusted_classes(processed_data['prob_spam'], t)
    processed_data['pos_neg'] = processed_data.apply(posivite_negative, axis=1)
    thresholds, roc_auc_score, precision_score, recall_score, accuracy, conf_matrix = get_metrcis('adj_class')[4], get_metrcis('adj_class')[5], \
                                                                                      get_metrcis('adj_class')[6], get_metrcis('adj_class')[7], \
                                                                                      get_metrcis('adj_class')[8], get_metrcis('adj_class')[9]
    precision, recall = get_metrcis('adj_class')[2], get_metrcis('adj_class')[3]
    plot_two()
    st.markdown('**AUC score**: {:.2f} | **Precision score**: {:.2f} | **Recall score**: {:.2f}'.format(roc_auc_score,
                                                                                                        precision_score,
                                                                                                        recall_score))
    styled_table = conf_matrix.style.applymap(color_table)
    st.dataframe(styled_table)

    st.write("As a performance metrics we use the following combinations of the correct results and errors")

    st.markdown(r'''$\textcolor{orange}{TPR (recall)} = \dfrac{\textcolor{red}{TP}}{\textcolor{red}{TP}+\textcolor{#00A0FC}{FN}}$ |
                    $\textcolor{blue}{FPR} = \dfrac{\textcolor{blue}{FP}}{\textcolor{blue}{FP}+\textcolor{orange}{TN}}$ |
                    $\textcolor{red}{PPV (precision)} = \dfrac{\textcolor{red}{TP}}{\textcolor{red}{TP}+\textcolor{blue}{FP}}$  ''')

    st.markdown('**Recall or Sensitivity** answers this question: “Of all the messages that are truly spam, how many did we label correctly?”. '
                '**Precision** metric means “Of all spam messages that labeled as spam, how many are actually spam?”. '
                'The **false positive rate** (**FPR**) is the number of emails which are not spam but are identified as spam, ' 
                'divided by the total number of not spam messages.')


    fpr, tpr, precision, recall, thresholds = get_metrcis()[0], get_metrcis()[1], get_metrcis()[2], \
                                              get_metrcis()[3], get_metrcis()[4]

    roc_auc_score, precision_score, recall_score = get_metrcis('pred_class')[5], get_metrcis('pred_class')[6], get_metrcis('pred_class')[7]

    st.write('The main metric to evaluate the performance of the classifier is the **AUC ROC** . What stands for Area Under the Receiver '
             'Operating Characteristic Curve. The ROC curve plots the performance of a binary classifier under various threshold settings '
             'and measured by TPR and FPR. AUC represents degree or measure of classes separability. The higher the AUC, the better the model '
             'is at predicting Spam as Spam and Ham as Ham.')
    plot_one()
    st.markdown('**AUC**: {:.2f} | **Precision**: {:.2f} | **Recall**: {:.2f}'.format(roc_auc_score, precision_score, recall_score))

    #st.write("More metrics here")
    st.write("Check the Theory section to see what's going on under the hood of the algorithm in details.")

theory = st.checkbox('Theory')
if theory:
  st.write("**Bayes' Theory**")
  st.write("The main idea for our spam classification algorithm is to apply Bayes's theorem to sort class "
           "'Spam' out of 'Ham' (not spam). The algorithm itself belongs to the group of the supervised machine learning "
           "tasks for pattern classification when a training model is based on labeled training data which then can be used to assign "
           "a pre-defined class label to new objects. The whole mechanism of training and classification is based on the following rule "
           "formulated by Thomas Bayes:")

  st.markdown(r'''$\mathsf{PosteriorProbability} = \dfrac{\mathsf{\textcolor{red}{PriorProbability}}  \times  \mathsf{\textcolor{blue}{ConditionalProbability}}}{\mathsf{\textcolor{orange}{Evidence}}}$''')
  st.markdown("A posterior probability can be read as “What is the probability that a particular text composed of words($x_i$) belongs to class $S$ ?” "
              "and can be formulated as")
  st.markdown(r'''$P(S\mid x_i) = \dfrac{\textcolor{red}{P(S)} \times \textcolor{blue}{P(x_i\mid S)}}{\textcolor{orange}{P(x_i)}}$''')

  st.write('**Major assumption**')
  st.markdown('The words in text are conditionally independent of each other. Independence means that the probability of one observation does not affect the probability '
              'of another observation. Like when we are tossing the coin. That is why the Bayesian classifier also called naive algorithm because it tranform a coherent text into “a bag of words”. '
              'Following this assumption we can calculate conditional probabilities by using a *chain rule* and multiplying probabilities of individual words.')

  st.write('**Prior Probability**')
  st.write('A prior probability can be interpreted as the prior belief or *a priori* knowledge or “the general probability of encountering a particular class.” A prior probability can be estimated as')
  st.markdown(r'''$\textcolor{red}{P(S)} = \dfrac{N_s}{N_c}$''')
  st.markdown(r'''where
  + $N_s$: count of spam messages in training data
  + $N_c$: count of all messages in training data
  ''')

  st.write('**Conditional Probability**')
  st.markdown(r'''A conditional probability $\textcolor{blue}{P(x_i\mid S)}$ can be formulated as “How likely is it to observe this particular feature $x$ given that it belongs to class $S$ ?”''')
  st.markdown(r'A classic approach to estimate a conditional probability is to calculate the following frequency')
  st.markdown(r'''$\textcolor{blue}{P(x_i\mid S)} = \dfrac{N_{x_i,S}}{N_S}$''')
  st.markdown(r'''where
  + $N_{x_i,S}$ is the number of times feature $x_i$ (word) appears in samples from class $S$
  + $N_S$ is the total count of all features in class $S$
  ''')
  st.write("But a classic approach doesn't help us to deal with the words out of the training dataset. That is why an alternative method to "
           "estimate a conditional probability looks like")
  st.markdown(r'''$\textcolor{blue}{\hat{P}(x_i\mid S)} = \dfrac{\sum tf(x_i, d \in S) + \mathit{\alpha}}{\sum N_{d \in S} + \mathit{\alpha} \cdot V}$''')
  st.markdown(r'''where
  + $x_i$: A word from the feature vector $x$ of a particular sample.
  + $\sum tf(x_i, d \in S)$: The sum of raw term frequencies of word $x_i$ from all documents in the training sample that belong to class $S$
  + $\sum N_{d \in S}$: The sum of all term frequencies in the training dataset for class $S$
  + $\mathit{\alpha}$: An additive smoothing parameter ($\mathit{\alpha}=1$ for Laplace smoothing)
  + $V$: The size of the vocabulary (number of different words in the training set)
  ''')

  st.markdown(r'''The posterior probability of the given text can be calculated as a product of the probabilities of the individual words
   under the following *chain rule*:''')
  st.markdown(r'''$\textcolor{blue}{\hat{P}(x_i\mid S)} = P(x_1\mid S)\times P(x_2\mid S)\times ... \times P(x_d\mid S) = \displaystyle\prod_{i=1}^d P(x_i\mid S)$''')

  st.write('**Evidence**')
  st.markdown(r'''The evidence $P(x)$ can be understood as the probability of encountering a particular pattern $x$ independent from the class label.
  Although the evidence term is required to accurately calculate the posterior probabilities, it can be removed since it is merely a scaling factor when
  we compare posterior probabilities of two classes''')
  st.markdown(r'''$\dfrac{P(x_i\mid S)\times P(S)}{P(x_i)} \gt \dfrac{P(x_i\mid H)\times P(H)}{P(x_i)} \implies P(x_i\mid S)\times P(S) \gt P(x_i\mid H)\times P(H)$ ''')

  st.write('**Floating point underflow**')
  st.markdown('Multiplying a set of small probabilities together will probably result in '
              '*floating point underflow* where the product will become too small to represent and will be replaced by 0. Instead of estimating ')
  st.markdown(r'''$P(S)\displaystyle\prod_{i=1}^nP(x_i\mid S)$, we consider computing the logarithm of''')
  st.markdown(r'''$\log(P(S)\displaystyle\prod_{i=1}^nP(x_i\mid S))$ which can be written equivalently as''')
  st.markdown(r'''$\log(P(S))+\displaystyle\sum_{i=1}^n \log(P(x_i\mid S))$''')
  st.write('Which can be used for final decision on how the text should be classified')
  st.markdown(r'''$\log(P(S))+\displaystyle\sum_{i=1}^n \log(P(x_i\mid S))\gt\log(P(H))+\displaystyle\sum_{i=1}^n \log(P(x_i\mid H))$''')
  st.markdown('The text should be assigned to that class which has a greater sum of logarithms.')

  st.markdown("**Maximum a posteriori estimation (MAP)**")
  st.markdown("Technically the probability shown at the beginning is not 100% correct. In order to classify the given text "
              'the concept of maximum a posteriori estimation was employed, which in turn closely related to the '
              'maximum likelihood estimation. Mathematically likelihood and probability are '
              'two different concepts, but I use likelihood to show how likely the given text belongs to class $S$. '
              'By calculating MAP we maximize a posterior probability. What means we are taking into account only prior and '
              "conditional probability without calculating the evidence")

  st.markdown("In case we'd like to transform log values into decimal or percent form the following formulas can be used:")
  st.markdown(r'''$S_{map} = \dfrac{1}{1+e^{q_h - q_s}}$''')
  st.markdown(r'''$H_{map} = \dfrac{1}{1+e^{q_s - q_h}}$''')
  st.markdown(r'''where $q_s$ and $q_h$ is a sum of logarithms of a prior and conditional probability for $S$ and $H$ class respectively''')
  st.write("Having MAP calculated in decimal form we can set up a custom threshold from 0 to 1 to sort out classes as shown in the 'Algorithm' section above. ")
  st.write("Check the source code below to explore how the theory is implemented")

  link_one = make_clickable('https://sebastianraschka.com/Articles/2014_naive_bayes_1.html', 'post')
  link_two = make_clickable('https://courses.cs.washington.edu/courses/cse312/18sp/lectures/naive-bayes/naivebayesnotes.pdf', 'here')
  st.write('All the credits go to Sebastian Rashka who beatifully put all the theory into one {}. I just added from {} log'.format(link_one, link_two), unsafe_allow_html=True)

code = st.checkbox('Source code')
if code:
  with st.echo():
      class SpamClassifier():
          def __init__(self):
              pass

          def fit(self, train):

              # words_frequency
              self.spam_dict = {}
              self.ham_dict = {}

              train['email'] = train['email'].astype('str')
              for i in range(0, train.shape[0]):
                  if train['label'][i] == 1:
                      for j in train['email'][i].split(' '):
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
              if label == 'SPAM':
                  if word.lower() in self.spam_dict:
                      cond_word_spam = np.log(
                          (k + self.spam_dict[word.lower()]) / (len(self.unique_words) + sum(self.spam_dict.values())))
                  else:
                      cond_word_spam = np.log((k) / (len(self.unique_words) + sum(self.spam_dict.values())))

                  self.cond_word_spam = cond_word_spam
                  return self.cond_word_spam

              else:
                  if word.lower() in self.ham_dict:
                      cond_word_ham = np.log((k + self.ham_dict[word.lower()]) / (
                              len(self.unique_words) + sum(self.ham_dict.values())))
                  else:
                      cond_word_ham = np.log((k) / (len(self.unique_words) + sum(self.ham_dict.values())))
                  self.cond_word_ham = cond_word_ham
                  return self.cond_word_ham

          # conditional probability for text, chain rule implementation
          def conditional(self, text, label):
              if label == 'SPAM':
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

          def classify(self, text, prob=False):
              ham = np.log(self.prior_ham) + SpamClassifier.conditional(text, 'NOT_SPAM')
              spam = np.log(self.prior_spam) + SpamClassifier.conditional(text, 'SPAM')

              while prob == True:
                  prob_ham = 1 / (1 + np.exp(spam - ham))
                  prob_spam = 1 / (1 + np.exp(ham - spam))
                  return prob_spam
                  break
              if spam > ham:
                  return 1
              else:
                  return 0


# streamlit run spam-detector.py
