#imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
#defing dataframe
df = pd.read_csv(r"Housing.csv")
#cleaning nonbinary entries
df['mainroad'] = df['mainroad'].replace("yes", 1)
df['mainroad'] = df['mainroad'].replace("no", 0)
df['guestroom'] = df['guestroom'].replace("yes", 1)
df['guestroom'] = df['guestroom'].replace("no", 0)
df['basement'] = df['basement'].replace("yes", 1)
df['basement'] = df['basement'].replace("no", 0)
df['hotwaterheating'] = df['hotwaterheating'].replace("yes", 1)
df['hotwaterheating'] = df['hotwaterheating'].replace("no", 0)
df['airconditioning'] = df['airconditioning'].replace("yes", 1)
df['airconditioning'] = df['airconditioning'].replace("no", 0)
df['prefarea'] = df['prefarea'].replace("yes", 1)
df['prefarea'] = df['prefarea'].replace("no", 0)
df['furnishingstatus'] = df['furnishingstatus'].replace("unfurnished", 0)
df['furnishingstatus'] = df['furnishingstatus'].replace("semi-furnished", 0.5)
df['furnishingstatus'] = df['furnishingstatus'].replace("furnished", 1)
#defining x to be the dataset other than the value that is to be reserved for y and predicted
X = df.drop('price', axis=1)
y = df['price']
#setting up model fields
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_actual_data = y_test
#preprocessing the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#setting up the linearregression model
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
#evaluating the accuracy of the model with the R2-Score with 1 being a perfectly accurate model
def model_evaluation(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    R2_Score = metrics.r2_score(y_test, y_pred)
    
    return pd.DataFrame([R2_Score], index=['R2-Score'], columns=[model_name])
model_evaluation(linear_reg, X_test_scaled, y_test, 'Linear Reg.')
