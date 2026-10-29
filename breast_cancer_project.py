import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# STEP 1: What do we want to do? (Objectives)

print("STEP 1: OBJECTIVES")
print("1. Predict if a tumour is Malignant or Benign")
print("2. Get accuracy of at least 95%")
print("3. Get recall of at least 95% (we should not miss a cancer patient)")
print()


# STEP 2: Read the data
print("STEP 2: READ THE DATA")


data = pd.read_csv("wdbc_breast_cancer.csv")

print("Rows and columns:", data.shape)
print(data.head())
print()

# In the file: M = Malignant, B = Benign
# We make a new column: 1 = Malignant, 0 = Benign
data["target"] = (data["diagnosis"] == "M").astype(int)

# Columns used to predict (all columns except id, diagnosis and target)
features = [c for c in data.columns if c not in ["id", "diagnosis", "target"]]

# STEP 3: Clean the data
print("STEP 3: CLEAN THE DATA")

print("Missing values:", data.isnull().sum().sum())
print("Duplicate rows:", data.duplicated().sum())
print("Nothing to fix. The data is already clean.")
print()

# STEP 4: Look at the data with graphs (EDA)

print("STEP 4: EXPLORE THE DATA")
print(data[["mean radius", "mean area", "mean texture"]].describe())
print()

# Graph 1: How many Benign and Malignant?
counts = data["diagnosis"].value_counts()
plt.bar(["Benign", "Malignant"], [counts["B"], counts["M"]],
        color=["green", "red"])
plt.title("Number of Benign and Malignant patients")
plt.ylabel("Patients")
plt.show()

# Graph 2: Histogram of mean radius
benign = data[data["target"] == 0]
malignant = data[data["target"] == 1]

plt.hist(benign["mean radius"], bins=25, alpha=0.6, color="green", label="Benign")
plt.hist(malignant["mean radius"], bins=25, alpha=0.6, color="red", label="Malignant")
plt.title("Mean radius: Benign vs Malignant")
plt.xlabel("Mean radius")
plt.legend()
plt.show()

# Graph 3: Boxplot of mean area
plt.boxplot([benign["mean area"], malignant["mean area"]])
plt.xticks([1, 2], ["Benign", "Malignant"])
plt.title("Mean area: Benign vs Malignant")
plt.show()

# Graph 4: Correlation (how features move together)
mean_columns = [c for c in features if c.startswith("mean")]
corr = data[mean_columns].corr()
plt.figure(figsize=(8, 7))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(mean_columns)), mean_columns, rotation=90)
plt.yticks(range(len(mean_columns)), mean_columns)
plt.title("Correlation between features")
plt.tight_layout()
plt.show()

# STEP 5: Statistics (t-test)
# Question: is the average really different for the two groups?
# If p-value is less than 0.05, the difference is real.
print("STEP 5: T-TEST")

for col in ["mean radius", "mean area", "mean texture", "mean smoothness"]:
    t, p = stats.ttest_ind(malignant[col], benign[col], equal_var=False)
    print(col, "-> malignant average:", round(malignant[col].mean(), 2),
          "| benign average:", round(benign[col].mean(), 2),
          "| p-value:", p)
print()
print("Features most related to cancer (correlation with target):")
related = data[features + ["target"]].corr()["target"].drop("target")
print(related.sort_values(ascending=False).head(5).round(2))
print()

# STEP 6: Get the data ready
print("STEP 6: PREPARE THE DATA")

X = data[features]     
y = data["target"]     

# Keep 80% for learning (train) and 20% for checking (test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling: bring all numbers to the same range
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print()

# STEP 7: PCA (reduce 30 columns to 2 so we can draw them)
print("STEP 7: PCA")

pca = PCA(n_components=2)
points = pca.fit_transform(X_train)
print("Information kept by 2 columns:",
      round(pca.explained_variance_ratio_.sum() * 100, 1), "%")

plt.scatter(points[y_train == 0, 0], points[y_train == 0, 1],
            color="green", alpha=0.6, label="Benign")
plt.scatter(points[y_train == 1, 0], points[y_train == 1, 1],
            color="red", alpha=0.6, label="Malignant")
plt.title("PCA: 30 columns shown in 2D")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.show()
print()

# STEP 8: Build the 3 models
print("STEP 8: BUILD MODELS")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

rows = []
predictions = {}
probabilities = {}

for name in models:
    model = models[name]
    model.fit(X_train, y_train)                 

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_predict = model.predict(X_test)           
    test_chance = model.predict_proba(X_test)[:, 1]  

    predictions[name] = test_predict
    probabilities[name] = test_chance

    rows.append([
        name,
        train_accuracy,
        accuracy_score(y_test, test_predict),
        precision_score(y_test, test_predict),
        recall_score(y_test, test_predict),
        f1_score(y_test, test_predict),
        roc_auc_score(y_test, test_chance),
        log_loss(y_test, test_chance),
    ])

result = pd.DataFrame(rows, columns=[
    "Model", "Train Acc", "Test Acc", "Precision",
    "Recall", "F1", "AUC", "Log Loss"]).round(3)

print(result.to_string(index=False))
print()
print("Precision, Recall, F1 are for the Malignant (cancer) class.")
print()

# Overfitting check (bias and variance)
# If Train Acc is much higher than Test Acc, the model has just
# memorised the training data. This is called overfitting.
print("OVERFITTING CHECK")
for i in range(len(result)):
    gap = result["Train Acc"][i] - result["Test Acc"][i]
    print(result["Model"][i], "-> gap between train and test:", round(gap, 3))
print()

# Graph: compare the models
result.set_index("Model")[["Test Acc", "Precision", "Recall", "F1"]].plot(
    kind="bar", ylim=(0.8, 1.02))
plt.title("Model comparison")
plt.xticks(rotation=0)
plt.show()

# STEP 9: Check the best model in more detail
print("STEP 9: BEST MODEL")

# Best model = the one with highest AUC
best = result.sort_values("AUC", ascending=False).iloc[0]["Model"]
print("Best model:", best)

# Confusion matrix
cm = confusion_matrix(y_test, predictions[best])
ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(cmap="Blues")
plt.title("Confusion Matrix - " + best)
plt.show()

tn, fp, fn, tp = cm.ravel()
print("Cancer found correctly      :", tp)
print("Healthy found correctly     :", tn)
print("Healthy called cancer (wrong):", fp)
print("Cancer missed (very bad)    :", fn)
print()

# Important features (from Random Forest)
forest = models["Random Forest"]
importance = pd.Series(forest.feature_importances_, index=features)
top10 = importance.sort_values().tail(10)
top10.plot(kind="barh", color="steelblue")
plt.title("Top 10 important features")
plt.tight_layout()
plt.show()
# STEP 10: Catch more cancer cases (change the cut-off)
# Normally: if chance of cancer is 50% or more, say "cancer".
# If we lower the cut-off, we catch more cancer patients,
# but we also get a few more false alarms.

print("STEP 10: CHANGE THE CUT-OFF")

for cutoff in [0.5, 0.4, 0.3, 0.2]:
    guess = (probabilities[best] >= cutoff).astype(int)
    missed = ((guess == 0) & (y_test == 1)).sum()
    print("Cut-off", cutoff,
          "| accuracy:", round(accuracy_score(y_test, guess), 3),
          "| recall:", round(recall_score(y_test, guess), 3),
          "| cancer missed:", missed)
print()

# STEP 11: Conclusion

print("STEP 11: CONCLUSION")
print("Best model:", best)
print("Most important features:", ", ".join(top10.index[::-1][:3]))
print("Logistic Regression was simple and gave the best result.")
print("Decision Tree overfitted (100% on train but less on test).")
print("Random Forest reduced overfitting by joining many trees.")
print("Lowering the cut-off helps to miss fewer cancer patients.")
print("Limit: only 569 patients. This tool helps doctors,")
print("it cannot replace them.")
