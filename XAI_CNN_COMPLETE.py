# Libraries for all the tasks
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report
from captum.attr import DeepLift
import shap
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    # Seed and find device
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Connect to GPU
    print(f"Using device: {device}")
    print(torch.cuda.is_available())     

    
    # ==================================================================
    # Importing data
    # ==================================================================
    print("\nIMPORTING DATA")

    # Change path to own directory
    data = pd.read_csv(r"C:\...\imu-hand-tremor-parkinsons.csv")
    print(data.head(5))

    # Acquire X and y
    X = data.drop(['label'],axis=1)
    y = data['label']

    # Make a train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y)
    

    # ==================================================================
    # Inspecting data
    # ==================================================================
    print("\nDATA INSPECTION")

    # Counter to inspect balance
    counter = Counter(y)

    plt.figure(figsize=(8,8))
    plt.bar(counter.keys(),counter.values(), color="dodgerblue")
    plt.title("Occurences of each label in the data")
    plt.xticks([1,2,3])
    plt.xlabel('Labels')
    plt.ylabel('Occurences')
    plt.show()



    # ==================================================================
    # Preprocessing
    # ==================================================================
    print("\nPREPROCESSING PHASE")

    # Apply a scaler
    scaler      = StandardScaler()
    X_train  = scaler.fit_transform(X_train)
    X_test   = scaler.transform(X_test)
    
    # PyTorch tensors
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train.to_numpy(), dtype=torch.long) - 1
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_te_t = torch.tensor(y_test.to_numpy(),  dtype=torch.long) - 1

    # Converting to DataLoader
    train_dl = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)



    # ==================================================================
    # Small model
    # ==================================================================
    print("\nINITIALISING MODEL")

    # The MLP
    class Model(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()

            # Network definition
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),  nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64,     32),  nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, out_dim)
            )

        # Forward pass
        def forward(self, x):
            return self.net(x)
    
    # Defining dimensions input and output
    in_dim = X_tr_t.shape[1] 
    out_dim = 3

    # Calling the model using GPU
    model = Model(in_dim, out_dim).to(device)

    # Definition of criterion and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    

    # ==================================================================
    # Training
    # ==================================================================
    print("\nTRAINING PHASE")

    # Defining training variables
    epoch_losses = []
    epoch_accuracies = []
    batch_size = 64
    epoch_nr = 30

    # Loop over epochs
    for epoch in range(epoch_nr):
        model.train()

        train_loss = 0
        train_correct = 0
        train_total = 0

        # looping over the batches
        for X_batch, y_batch in train_dl:

            # Putting to GPU
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Model training
            optimizer.zero_grad()
            logits  = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            # Saving informative variables
            train_loss += loss.item() * len(X_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(X_batch)

        # Update variables
        epoch_losses.append(train_loss / train_total)
        epoch_accuracies.append(train_correct / train_total)

    # Save model if initialising again
    # torch.save(model.state_dict(), 'saved_model.pth')

    # Load trained model        
    model = Model(in_dim=6, out_dim=3).to(device)
    model.load_state_dict(torch.load("saved_model.pth", map_location=device, weights_only=True))
    model.eval()

    # Plot the training statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epoch_losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(epoch_accuracies, color='green')
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    plt.show()


    # ==================================================================
    # Evaluation
    # ==================================================================

    print("\nEVALUATION PHASE")

    # Set model to evaluation mode
    model.eval()

    # Disable gradients and predict
    with torch.no_grad():
        X_te_t = X_te_t.to(device)
        outputs    = model(X_te_t)
        y_pred_nn = outputs.argmax(dim=1).cpu().numpy()  

    # Statistics
    print(f"Accuracy:  {accuracy_score(y_te_t.numpy(), y_pred_nn):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_te_t.numpy(), y_pred_nn))

    # Load the model
    model.load_state_dict(torch.load("saved_model.pth", map_location=device, weights_only=True))
    model.eval()
    
    # =================================================
    # Shapley
    # =================================================

    print("\nSHAPLEY")
    # Definition of feature names
    feature_names = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    # Predict function for Shapley
    def predict(x):
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32).to(device)
            probs  = torch.softmax(model(tensor), dim=1)
        return probs.cpu().numpy()


    # SHAP explanations
    shap_expl = shap.KernelExplainer(predict, shap.kmeans(X_tr_t.numpy(), 10),  feature_names=feature_names)
    nb_points_explain = 400
    shap_values = shap_expl(X_tr_t.numpy()[0:nb_points_explain])
    
    # Find the reference
    ref = shap_expl.expected_value
    print("\nAverage predicted output: ", ref)

    # Second explanation class 1, i=0
    class_idx = 1
    i = 0
    fig = plt.figure()
    shap.plots.waterfall(shap_values[i,:,class_idx])
    plt.show()
    print("Corresponding label for sample 1: ", y_tr_t.data[0])

    # Second explanation class 1, i=1
    i = 1
    fig = plt.figure()
    shap.plots.waterfall(shap_values[i,:,class_idx])
    plt.show()
    print("Corresponding label for sample 2: ", y_tr_t.data[1])


    # =================================================
    # DeepLIFT
    # =================================================

    print("\nDEEPLIFT")

    # Deeplift explanations
    dl_expl      = DeepLift(model)
    baseline     = torch.zeros(1, 6).to(device)

    # Attributions class 1
    attributions = dl_expl.attribute(X_tr_t.to(device), baselines=baseline, target=1)
    attr_np      = attributions.detach().cpu().numpy() 

    # Local explanation class=1, i=0
    sorted_idx      = np.argsort(np.abs(attr_np[0]))          
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_values   = attr_np[0][sorted_idx]

    plt.figure(figsize=(8, 4))
    plt.barh(sorted_features, sorted_values, color='crimson')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title("DeepLIFT feature attributions for a single example")
    plt.xlabel("Attribution score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # Local explanation class=1,i=1
    sorted_idx      = np.argsort(np.abs(attr_np[1]))          
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_values   = attr_np[1][sorted_idx]

    plt.figure(figsize=(8, 4))
    plt.barh(sorted_features, sorted_values, color='crimson')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title("DeepLIFT feature attributions for a single example")
    plt.xlabel("Attribution score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()




    # =================================================
    # Comparison DeepLift & Shap
    # =================================================
    print("\nCOMPARISON")

    # Calculating mean absolute values single class
    mean_abs_shap = np.abs(shap_values.values[:, :, class_idx]).mean(axis=0)  
    mean_abs_dl   = np.abs(attr_np).mean(axis=0) 

    # Normalise values for comparisons
    shap_norm = mean_abs_shap / mean_abs_shap.max()
    dl_norm   = mean_abs_dl   / mean_abs_dl.max()

    # Viarables useful for plotting
    x = np.arange(len(feature_names))
    w = 0.35

    # Comparison plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w/2, shap_norm, w, label='SHAP explanation per feature normalised',     color='dodgerblue')
    ax.bar(x + w/2, dl_norm,   w, label='DeepLIFT explanation per feature normalised', color='crimson')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.set_ylabel("Normalised Importance")
    ax.set_title("SHAP vs DeepLIFT on feature importance comparison")
    ax.legend()
    plt.show()


    # Most important features 
    feature_imps_shap = [x for _, x in sorted(zip(mean_abs_shap, feature_names), reverse=True)]  
    feature_imps_dl = [x for _, x in sorted(zip(mean_abs_dl, feature_names), reverse=True)]                             
    print("mean absolute shapley values: ", feature_imps_shap)
    print("mean absolute DeepLift values: ", feature_imps_dl)

    # Most important feature
    print("Top SHAP feature:    ", feature_imps_shap[0])
    print("Top DeepLIFT feature:", feature_imps_dl[0])

    # Comparison on agreement usign spearman
    from scipy.stats import spearmanr
    rho, p = spearmanr(mean_abs_shap, mean_abs_dl)
    print(f"Spearman rank correlation: ρ={rho:.4f}, p={p:.4f}")




