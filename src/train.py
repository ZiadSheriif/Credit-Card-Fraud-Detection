from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class Trainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train(self):
        self.model = SVC()
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        return self.model.score(self.X_test, self.y_test)
        
#! I think we will use SVM , Decision Tree and Random Forest