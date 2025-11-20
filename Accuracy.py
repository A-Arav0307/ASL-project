import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def acc_det(data, name):
    print("testing")
    df = pd.read_csv(data)
    df = df.set_index('Epoch')
    
    # Create figure with 4 subplots (2x2)
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Model Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['accuracy'], label='Train Accuracy', marker='o', linestyle='-')
    plt.plot(df.index, df['val_accuracy'], label='Val Accuracy', marker='x', linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Model Loss
    plt.subplot(2, 2, 2)
    plt.plot(df.index, df['loss'], label='Train Loss', marker='o', linestyle='-')
    plt.plot(df.index, df['val_loss'], label='Val Loss', marker='x', linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Prepare data for linear regression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, root_mean_squared_error
    
    epochs = df.index.values.reshape(-1, 1)
    regression_stats = {}
    
    # Subplot 3: Linearized Accuracy with Linear Regression (Inverse Transform)
    plt.subplot(2, 2, 3)
    
    # Train Accuracy - inverse transform for saturation curve: 1/y vs x
    train_acc = df['accuracy'].values.reshape(-1, 1)
    train_acc_inv = 1 / train_acc
    model_train_acc = LinearRegression()
    model_train_acc.fit(epochs, train_acc_inv)
    train_acc_inv_pred = model_train_acc.predict(epochs)
    
    plt.scatter(df.index, train_acc_inv, label='1/(Train Accuracy)', marker='o', alpha=0.6)
    plt.plot(df.index, train_acc_inv_pred, label='Train Accuracy Linear Fit', linestyle='-', linewidth=2)
    
    # Val Accuracy - inverse transform
    val_acc = df['val_accuracy'].values.reshape(-1, 1)
    val_acc_inv = 1 / val_acc
    model_val_acc = LinearRegression()
    model_val_acc.fit(epochs, val_acc_inv)
    val_acc_inv_pred = model_val_acc.predict(epochs)
    
    plt.scatter(df.index, val_acc_inv, label='1/(Val Accuracy)', marker='x', alpha=0.6)
    plt.plot(df.index, val_acc_inv_pred, label='Val Accuracy Linear Fit', linestyle='--', linewidth=2)
    
    plt.title('Linearized Accuracy (Inverse Transform)')
    plt.xlabel('Epoch')
    plt.ylabel('1/(Accuracy)')
    plt.legend()
    plt.grid(True)
    
    # Store accuracy regression stats
    regression_stats['train_accuracy'] = {
        'slope': model_train_acc.coef_[0][0],
        'intercept': model_train_acc.intercept_[0],
        'r2_score': r2_score(train_acc_inv, train_acc_inv_pred),
        'rmse': root_mean_squared_error(train_acc_inv, train_acc_inv_pred),
        'predicted_limit': 1 / model_train_acc.intercept_[0] if model_train_acc.intercept_[0] > 0 else None
    }
    
    regression_stats['val_accuracy'] = {
        'slope': model_val_acc.coef_[0][0],
        'intercept': model_val_acc.intercept_[0],
        'r2_score': r2_score(val_acc_inv, val_acc_inv_pred),
        'rmse': root_mean_squared_error(val_acc_inv, val_acc_inv_pred),
        'predicted_limit': 1 / model_val_acc.intercept_[0] if model_val_acc.intercept_[0] > 0 else None
    }
    
    # Subplot 4: Linearized Loss with Linear Regression (Log Transform)
    plt.subplot(2, 2, 4)
    
    # Train Loss - log transform for exponential decay
    train_loss = df['loss'].values.reshape(-1, 1)
    train_loss_log = np.log(train_loss)
    model_train_loss = LinearRegression()
    model_train_loss.fit(epochs, train_loss_log)
    train_loss_log_pred = model_train_loss.predict(epochs)
    
    plt.scatter(df.index, train_loss_log, label='log(Train Loss)', marker='o', alpha=0.6)
    plt.plot(df.index, train_loss_log_pred, label='Train Loss Linear Fit', linestyle='-', linewidth=2)
    
    # Val Loss - log transform
    val_loss = df['val_loss'].values.reshape(-1, 1)
    val_loss_log = np.log(val_loss)
    model_val_loss = LinearRegression()
    model_val_loss.fit(epochs, val_loss_log)
    val_loss_log_pred = model_val_loss.predict(epochs)
    
    plt.scatter(df.index, val_loss_log, label='log(Val Loss)', marker='x', alpha=0.6)
    plt.plot(df.index, val_loss_log_pred, label='Val Loss Linear Fit', linestyle='--', linewidth=2)
    
    plt.title('Linearized Loss (Log Transform)')
    plt.xlabel('Epoch')
    plt.ylabel('log(Loss)')
    plt.legend()
    plt.grid(True)
    
    # Store loss regression stats
    regression_stats['train_loss'] = {
        'slope': model_train_loss.coef_[0][0],
        'intercept': model_train_loss.intercept_[0],
        'r2_score': r2_score(train_loss_log, train_loss_log_pred),
        'rmse': root_mean_squared_error(train_loss_log, train_loss_log_pred),
        'decay_rate': -model_train_loss.coef_[0][0]
    }
    
    regression_stats['val_loss'] = {
        'slope': model_val_loss.coef_[0][0],
        'intercept': model_val_loss.intercept_[0],
        'r2_score': r2_score(val_loss_log, val_loss_log_pred),
        'rmse': root_mean_squared_error(val_loss_log, val_loss_log_pred),
        'decay_rate': -model_val_loss.coef_[0][0]
    }
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(name)
    
    return regression_stats