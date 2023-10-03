import matplotlib.pyplot as plt
import numpy as np

# Updating the models and accuracies with the ResNet-50 results
models = [
    "1-Layer CNN",
    "LeNet-5",
    "VGG-16",
    "VGG-16 + IN weights",
    "ResNet-50"
]

datasets = ["MNIST", "CIFAR-10", "Tiny ImageNet"]

# Updated accuracies with ResNet-50 results
accuracies = [
    [0.9852, 0.6343, 0.005],  # 1-Layer CNN
    [0, 0, 0.1737],  # LeNet-5
    [0.9941, 0.1, 0.005],  # VGG-16
    [0, 0.7433, 0.4229],  # VGG-16 + IN weights
    [0.9932, 0.8039, 0.2729]  # ResNet-50
]

x = np.arange(len(datasets))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))

rects = []
for i, (model, acc) in enumerate(zip(models, accuracies)):
    rects.append(ax.bar(x + i*width, acc, width, label=model))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy by Dataset and Model Configuration')
ax.set_xticks(x + width*(len(models)/2 - 0.5))
ax.set_xticklabels(datasets)
ax.legend()

# Autolabel function to display the label on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 4)) if height is not None else '',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

for rect in rects:
    autolabel(rect)

fig.tight_layout()
plt.xticks(rotation=0)  # Making labels along the x-axis straight
plt.ylim([0, 1.1])  # to better visualize the accuracies
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('paper/accuracy_comparison.png', bbox_inches='tight', dpi=300)
