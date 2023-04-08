import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
from sklearn import preprocessing
# Read csv data file
df = pd.read_csv('C:\\Users\\Serene Ooi\\OneDrive\\Documents\\Year 2 Semester 1\\BACS3013 - Data Science\\titanic.csv')
# Drop attribute Name
# Setting axis=1 means we want to drop a column
df = df.drop('Name', axis = 1)
# Drop PassengerId Ticket, Cabin and Embarked
df = df.drop(['PassengerId', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

# Drop the null value rows. 
df.dropna(inplace=True)
df.isnull().sum()

# Convert integer to string: survived
df['Survived'] = df['Survived'].astype(str)
df['Survived'].describe()

# Convert integer to string: pclass
df['Pclass'] = df['Pclass'].astype(str)
# Convert integer to string: sex
df['Sex'] = df['Sex'].astype(str)
df.dtypes

# Import LabelEncoder
from sklearn import preprocessing
# Create LabelEncoder
le = preprocessing.LabelEncoder()
# Convert string categories into numbers for sex
df['Sex'] = le.fit_transform(df['Sex'])
df.head()

# Indicate the target column
target = df['Survived']

# Indicate the columns that will serve as features
features = df.drop('Survived', axis = 1)

names = features.columns

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit the data on the Scaler object
scaled_features = scaler.fit_transform(features)

# After standardization, scaled_features is transformed into an array so we need to convert
scaled_features = pd.DataFrame(scaled_features, columns = names)
scaled_features.head()
# Split data into train, validation and test sets
# Import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# Split the dataset into training + development set and test set
x, x_test, y, y_test = train_test_split(scaled_features, target, test_size = 0.2, random_state = 0)

# Split the dataset into training set and development set
x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size = 0.2, random_state = 10)
# Create a KNN classifier with k = 3
knn = KNeighborsClassifier(n_neighbors = 3)

# Train the model using the training set
knn.fit(x_train, y_train)

# Predict the target for the development dataset
dev_predict = knn.predict(x_dev)

# Compute the model accuracy on the development set: How often is the classifier correct?
print("Accuracy (Dev): ", metrics.accuracy_score(y_dev, dev_predict))
# Predict the target for the test dataset
test_predict = knn.predict(x_test)

# Compute the model accuracy on the development set: How often is the classifier correct?
print("Accuracy (Test): ", metrics.accuracy_score(y_test, test_predict))

clf = tree.DecisionTreeClassifier()

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=5, random_state = 0)

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
dev_predict = clf.predict(x_dev)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_dev, dev_predict)) 

text_representation = tree.export_text(clf)
print(text_representation)