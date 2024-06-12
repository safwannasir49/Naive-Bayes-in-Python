<h2 align="center">Naive Baye's Classifier</h2>
<p>Naïve Bayes classification is a straightforward and powerful algorithm for the classification task. Naïve Bayes classification is based on applying Bayes’ theorem with a strong independence assumption between the features. Naïve Bayes classification produces good results when used for textual data analysis such as Natural Language Processing.</p>
<p>Naïve Bayes models are also known as simple Bayes or independent Bayes. All these names refer to the application of Bayes’ theorem in the classifier’s decision rule. Naïve Bayes classifier applies the Bayes’ theorem in practice. This classifier brings the power of Bayes’ theorem to machine learning.</p>
<h2>Naive Bayes Algorithm Intuition</h2>
<p>Naïve Bayes Classifier uses the Bayes’ theorem to predict membership probabilities for each class, such as the probability that a given record or data point belongs to a particular class. The class with the highest probability is considered the most likely class. This is also known as the Maximum A Posteriori (MAP).</p>
<p>The MAP for a hypothesis with 2 events A and B is:</p>
<pre>
MAP (A)
= max (P (A | B))
= max (P (B | A) * P (A))/P (B)
= max (P (B | A) * P (A))
</pre>
<p>Here, P(B) is the evidence probability. It is used to normalize the result. It remains the same, so removing it would not affect the result.</p>
<p>Naïve Bayes Classifier assumes that all the features are unrelated to each other. The presence or absence of a feature does not influence the presence or absence of any other feature.</p>
<p>In real-world datasets, we test a hypothesis given multiple evidence on features, making the calculations quite complicated. To simplify the work, the feature independence approach is used to uncouple multiple evidence and treat each as an independent one.</p>
<h2>Types of Naïve Bayes Algorithm</h2>
<p>The 3 types are listed below:</p>
<ul>
    <li>Gaussian Naïve Bayes</li>
    <li>Multinomial Naïve Bayes</li>
    <li>Bernoulli Naïve Bayes</li>
</ul>
<h2>Applications of Naive Bayes</h2>
<p>Naïve Bayes is one of the most straightforward and fast classification algorithms. It is very well suited for large volumes of data. It is successfully used in various applications such as:</p>
<ul>
    <li>Spam filtering</li>
    <li>Text classification</li>
    <li>Sentiment analysis</li>
    <li>Recommender systems</li>
</ul>
<h2>Dataset</h2>
<p>Adult Dataset</p>
<h2>Implementation</h2>
<p><b>Libraries:</b> NumPy, pandas, matplotlib, sklearn, seaborn</p>
<h3>Data Exploration</h3>
<p>The categorical variables are:</p>
<pre>
['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']
</pre>
<p><b>Cardinality:</b> The number of labels within a categorical variable is known as cardinality. A high number of labels within a variable is known as high cardinality. High cardinality may pose some serious problems in the machine learning model.</p>
<pre>
for var in categorical:
    print(var, ' contains ', len(df[var].unique()), ' labels')
</pre>
<pre>
workclass  contains  9  labels
education  contains  16  labels
marital_status  contains  7  labels
occupation  contains  15  labels
relationship  contains  6  labels
race  contains  5  labels
sex  contains  2  labels
native_country  contains  42  labels
income  contains  2  labels
</pre>
<p>There are 6 numerical variables:</p>
<pre>
['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
</pre>
<h3>Model Training, Evaluations, and Predictions</h3>
<pre>
from sklearn.naive_bayes import GaussianNB
instantiate the model
gnb = GaussianNB()

fit the model
gnb.fit(X_train, y_train)
</pre>

<p><b>Accuracy:</b></p>
<pre>
Training set score: 0.8067
Test set score: 0.8083
Null accuracy score: 0.7582
</pre>
<p>We can see that our model accuracy score is 0.8083 but the null accuracy score is 0.7582. So, we can conclude that our Gaussian Naive Bayes Classification model is doing a very good job in predicting the class labels.</p>
