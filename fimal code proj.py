
import pickle as pkl
model = pkl.load(open('spam_model','rb'))
cv = pkl.load(open('text_transform_model','rb'))

text = input("enter the testing email here")

email_text = [text]
numeric_data = cv.transform(email_text).toarray()


result = model.predict(numeric_data)
print(result[0])


#here in result if result value is 0 or ham then it is a valid email
# if value of result if 1 or spam them it is an invalid email
# if the value of result is 2 or social then it is an social email