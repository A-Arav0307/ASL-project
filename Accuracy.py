import pandas as pd
import matplotlib.pyplot as plt

def acc_det(data, name):
    print("testing")
    df = pd.read_csv(data)
    df = df.set_index('Epoch')
    plt.figure(figsize=(14, 6))

    # Subplot 1: Model Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(df.index, df['accuracy'], label='Train Accuracy', marker='o', linestyle='-')
    plt.plot(df.index, df['val_accuracy'], label='Val Accuracy', marker='x', linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Model Loss
    plt.subplot(1, 2, 2)
    plt.plot(df.index, df['loss'], label='Train Loss', marker='o', linestyle='-')
    plt.plot(df.index, df['val_loss'], label='Val Loss', marker='x', linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(name)