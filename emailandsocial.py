# importing used libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import pickle as pkl

# importing  data set and cleaning the data
pd2 = pd.read_csv("all.csv")
pd.set_option("display.max_columns", None)
print(pd2.shape)
arr = pd2.columns

data = pd2.drop(columns= arr[2:])
sns.heatmap(data.isnull())
plt.show()
print(data.shape)
data = data.dropna()
x = data["EmailText"]
y = data["Label"]

data = data.replace("ham", 0)
data = data.replace("spam", 1)
data = data.replace("social",2)
print(data.head)

MISSING =( data.isnull().sum()/data.shape[0])*100
print(MISSING)

df3 = data.dropna()
print(df3)
MISSING1 =(df3.isnull().sum()/df3.shape[0])*100
print(MISSING1)

arr = data.groupby("Label")
print(len(arr))
a = len(arr.get_group(0))
b = len(arr.get_group(1))
c = len(arr.get_group(2))
plot_data = [a, b, c]
plot_name = ["ham email", "spam email", "social email"]
plt.pie(plot_data, labels=plot_name, explode=[0.2, 0.3, 0], autopct="%0.2f%%", shadow=True)
plt.show()

#training and fitting model 1
cv = CountVectorizer()
xx = cv.fit_transform(x)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(xx, y, test_size=0.2, random_state=44)
print(x_test.shape, x_train.shape)
print(y_test.shape, y_train.shape)

# training and fitting model 2
model = MultinomialNB()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print("you can say you are getting a accuracy of {} percent this may change from dataset to dataset".format(acc * 100))

email = input("enter the email just for testing")
email_text = [email]
numeric_data = cv.transform(email_text).toarray()
results = model.predict(numeric_data)


print(results)
if results[0] == "ham":
    print("its a valid email")
if results[0] == "spam":
    print("its a spam email can contain virus and malware")
if results[0] == "social":
    print("its a social email")



#saving both the models 
pkl.dump(cv, open('text_transform_model', 'wb'))
pkl.dump(cv, open('spam_model', 'wb'))

