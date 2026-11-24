import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("dataset.csv")

print(data)

X = data[["StudyHours"]]
y = data["Marks"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

hours = float(input("Enter study hours: "))

new_data = pd.DataFrame({
    "StudyHours": [hours]
})

prediction = model.predict(new_data)

print("Predicted Marks:", round(prediction[0], 2))


plt.scatter(X, y, color="blue", label="Actual Data")

plt.plot(
    X.sort_values("StudyHours"),
    model.predict(X.sort_values("StudyHours")),
    color="red",
    label="Best Fit Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()


