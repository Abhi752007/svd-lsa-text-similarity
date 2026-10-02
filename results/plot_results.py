import pandas as pd
import matplotlib.pyplot as plt


# Load experiment results
df = pd.read_csv("results/precision_at_5.csv")


# --------------------------------
# Plot 1: Overall performance
# --------------------------------

overall = df[df["category"] == "Overall"].copy()

# Create method labels
overall["label"] = overall.apply(
    lambda row: (
        "TF-IDF"
        if row["method"] == "TF-IDF"
        else f"LSA k={int(row['k'])}"
    ),
    axis=1
)

plt.figure(figsize=(8, 5))

plt.plot(
    overall["label"],
    overall["precision_at_5"],
    marker="o"
)

plt.xlabel("Method")
plt.ylabel("Precision@5")
plt.title("Overall Precision@5")

plt.ylim(0, 1)
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/overall_precision.png",
    dpi=300
)

plt.show()


# --------------------------------
# Plot 2: Category comparison
# --------------------------------

category_df = df[df["category"] != "Overall"].copy()

# Create labels for the x-axis
category_order = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.med",
    "sci.space",
    "talk.politics.misc"
]

category_df["category"] = pd.Categorical(
    category_df["category"],
    categories=category_order,
    ordered=True
)

plt.figure(figsize=(10, 6))

for method in ["TF-IDF", "LSA"]:
    
    if method == "TF-IDF":
        data = category_df[
            category_df["method"] == "TF-IDF"
        ]
        
        plt.plot(
            data["category"],
            data["precision_at_5"],
            marker="o",
            label="TF-IDF"
        )

    else:
        for K in [20, 50, 100, 200]:
            data = category_df[
                (category_df["method"] == "LSA") &
                (category_df["k"] == K)
            ]

            plt.plot(
                data["category"],
                data["precision_at_5"],
                marker="o",
                label=f"LSA k={K}"
            )


plt.xlabel("Category")
plt.ylabel("Precision@5")
plt.title("Category-wise Precision@5")

plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/category_precision.png",
    dpi=300
)

plt.show()


print("Plots created successfully")
print("Saved:")
print("results/overall_precision.png")
print("results/category_precision.png")