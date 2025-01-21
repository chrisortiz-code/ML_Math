import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Generate randomized data
def generate_random_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        # Generate random wind (float between 0 and 1)
        wind = np.random.uniform(0, 1)

        # Generate a random distance with noise (0 to 100)
        distance = np.random.uniform(0, 100)

        # Decision logic with randomness
        # Example: Higher wind reduces the likelihood of kicking at greater distances
        kick_prob = max(0, 1 - (distance / 70) - (wind * 0.5))  # Wind adds a penalty to kicking probability
        kick = 1 if np.random.random() < kick_prob else 0

        # Append the result
        data.append({"Wind": wind, "Distance": distance, "Kick": kick})

    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Batch distances into 5-yard ranges
    df["Distance_Range"] = (df["Distance"] // 5 * 5).astype(int)

    return df

def generate_perfect_data(num_samples=500):
    data = []
    for _ in range(num_samples):
        wind = np.random.uniform(0, 1)
        distance = np.random.uniform(0, 100)
        if (distance <= 50 and wind <= 0.5) or (distance <= 30):
            kick = 1
        else:
            kick = 0
        data.append({"Wind": wind, "Distance": distance, "Kick": kick})
    return pd.DataFrame(data)

# Gini impurity calculation
def gini_index(groups, classes):
    total_samples = sum(len(group) for group in groups)
    gini = 0.0

    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = (group["Kick"] == class_val).sum() / size
            score += proportion ** 2
        gini += (1.0 - score) * (size / total_samples)

    return gini


# Split data
def split_data(data, feature, value):
    left = data[data[feature] <= value]
    right = data[data[feature] > value]
    return left, right


# Find the best split
def get_best_split(data):
    features = ["Wind", "Distance"]
    classes = data["Kick"].unique()
    best_gini = float("inf")
    best_split = None

    for feature in features:
        for value in data[feature].unique():
            groups = split_data(data, feature, value)
            gini = gini_index(groups, classes)
            if gini < best_gini:
                best_gini = gini
                best_split = {"feature": feature, "value": value, "groups": groups}

    return best_split


# Build the decision tree
def build_tree(data, max_depth, min_size, depth=0):
    if len(data) <= min_size or depth >= max_depth:
        return {"leaf": True, "prediction": data["Kick"].mode()[0]}

    split = get_best_split(data)
    if not split or len(split["groups"][0]) == 0 or len(split["groups"][1]) == 0:
        return {"leaf": True, "prediction": data["Kick"].mode()[0]}

    left, right = split["groups"]
    tree = {"leaf": False, "feature": split["feature"], "value": split["value"]}

    tree["left"] = build_tree(left, max_depth, min_size, depth + 1)
    tree["right"] = build_tree(right, max_depth, min_size, depth + 1)
    return tree


# Predict using the tree
def predict(tree, row):
    if tree["leaf"]:
        return tree["prediction"]
    if row[tree["feature"]] <= tree["value"]:
        return predict(tree["left"], row)
    else:
        return predict(tree["right"], row)


# Visualize the tree
def visualize_tree(tree, depth=0):
    if tree["leaf"]:
        print("\t" * depth + f"Predict: {tree['prediction']}")
    else:
        print("\t" * depth + f"[{tree['feature']} <= {tree['value']}]")
        visualize_tree(tree["left"], depth + 1)
        visualize_tree(tree["right"], depth + 1)


# Plot the data
def plot_data(data):
    kick_data = data[data["Kick"] == 1]
    no_kick_data = data[data["Kick"] == 0]

    plt.figure(figsize=(12, 6))
    plt.scatter(kick_data["Distance"], kick_data["Wind"], color="green", label="Kick", alpha=0.6, s=50,
                edgecolors="black")
    plt.scatter(no_kick_data["Distance"], no_kick_data["Wind"], color="red", label="No Kick", alpha=0.6, s=50,
                edgecolors="black")
    plt.title("Field Goal Decision Data with Continuous Wind", fontsize=16)
    plt.xlabel("Distance (0 to 100 yards)", fontsize=14)
    plt.ylabel("Wind (0 = Calm, 1 = Strong Wind)", fontsize=14)
    plt.xticks(range(0, 101, 10))
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


# Generate the data
# random_data = generate_random_data(num_samples=500)
# plot_data(random_data)
# decision_tree = build_tree(random_data, max_depth=4, min_size=10)

perfect_data = generate_perfect_data()
plot_data(perfect_data)
p_decision_tree = build_tree(perfect_data, max_depth=2, min_size=10)


# Visualize the decision tree
print("\nDecision Tree Structure:")
visualize_tree(p_decision_tree)

# Evaluate the decision tree
test_data = generate_random_data(num_samples=100)
test_data["Prediction"] = test_data.apply(lambda row: predict(p_decision_tree, row), axis=1)
accuracy = (test_data["Prediction"] == test_data["Kick"]).mean()
print(f"\nCustom Decision Tree Accuracy: {accuracy:.2f}")
