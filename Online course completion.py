import pandas as pd

df = pd.read_csv("course_completion_2000_dataset.csv")

print("Dataset Preview:")
print(df.head())

def categorize(value, low, high):
    if value < low:
        return "Low"
    elif value < high:
        return "Medium"
    else:
        return "High"

df["StudyHours"] = df["StudyHours"].apply(lambda x: categorize(x, 4, 7))
df["Attendance"] = df["Attendance"].apply(lambda x: categorize(x, 60, 80))
df["Assignment"] = df["Assignment"].apply(lambda x: categorize(x, 50, 75))

print("\nCategorized Data:")
print(df.head())

data = df.values.tolist()

def find_s(data):
    hypothesis = None
    for instance in data:
        if instance[-1] == "Complete":
            if hypothesis is None:
                hypothesis = instance[:-1]
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != instance[i]:
                        hypothesis[i] = "?"
    return hypothesis

def candidate_elimination(data):
    S = ['0'] * (len(data[0]) - 1)
    G = [['?'] * (len(data[0]) - 1)]
    
    for instance in data:
        if instance[-1] == "Complete":
            for i in range(len(S)):
                if S[i] == '0':
                    S[i] = instance[i]
                elif S[i] != instance[i]:
                    S[i] = '?'
            G = [g for g in G if all(
                g[i] == '?' or g[i] == instance[i] for i in range(len(g))
            )]
        else:
            new_G = []
            for g in G:
                for i in range(len(g)):
                    if g[i] == '?':
                        if S[i] != instance[i]:
                            new_h = g.copy()
                            new_h[i] = S[i]
                            new_G.append(new_h)
            G = new_G
    
    return S, G

S_find = find_s(data)
print("\nFind-S Hypothesis:", S_find)

S_final, G_final = candidate_elimination(data)

print("\nSpecific Hypothesis (S):", S_final)

print("\nGeneral Hypothesis (G):")
for g in G_final[:5]:
    print(g)

def predict(sample, S):
    for i in range(len(S)):
        if S[i] != '?' and S[i] != sample[i]:
            return "Not Complete"
    return "Complete"

test = ["High","High","High"]

print("\nTest Input:", test)
print("Prediction:", predict(test, S_find))
