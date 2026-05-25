import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\samik\OneDrive\Desktop\student.csv")

df['performance'] = (df['previous_marks'] >= 75).astype(int)

X = df[['study_hours', 'attendance', 'previous_marks']]
y = df['performance']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

student_names = df.iloc[y_test.index]['name']

results = pd.DataFrame({
    "Name": student_names,
    "Prediction": y_pred
})

results['Prediction'] = results['Prediction'].map({
    1: "Good Performance",
    0: "Needs Improvement"
})

print(results)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

study_hours = float(input("\nStudy Hours: "))
attendance = float(input("Attendance: "))
marks = float(input("Marks: "))

new_data = pd.DataFrame({
    'study_hours': [study_hours],
    'attendance': [attendance],
    'previous_marks': [marks]
})

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("\nGood Performance")
else:
    print("\nNeeds Improvement")
