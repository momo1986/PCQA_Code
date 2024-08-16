import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout()
        self.layer2 = nn.Linear(128,1)
    
    def forward(self, x):
        return self.layer2(self.dropout(self.bn(self.layer1(x)))) + 3.8638
model_names = [ "gluon_seresnext101_64x4d",'cait_m48_448','eca_nfnet_l1','openai/clip-vit-large-patch14-336', 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',"laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 'openai/clip-vit-base-patch32', 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K']
Xs =[]
mses =[]
for model_name in model_names:
    F = pd.read_parquet(model_name.replace('/','.') + '_train.parquet')
    if 'text_feature' in F.keys():
        X = np.array([np.concatenate([F.iloc[i].image_feature[0],F.iloc[i].text_feature[0]]) for i in tqdm(range(len(F)))])
    else:
        X = np.array([F.iloc[i].image_feature for i in tqdm(range(len(F)))])
    X = X/np.linalg.norm(X)
    Xs.append(X)
Xs = np.concatenate(Xs,1)
print(Xs.shape)
#X =  np.array([F.iloc[i].image_feature[0]  for i in range(len(F))])
Y = np.array([F.iloc[i].mos  for i in range(len(F))])
# orders = np.argsort(Y1)

# for i,odr in enumerate(orders):
#     Y1[odr] = i
# Y = Y1/14000.*5

for seed in range(1,18):

    seed_everything(seed)
    X_train, X_test, y_train, y_test = train_test_split(Xs, Y, test_size=0.2, random_state=seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
   
    input_dim = X_train.shape[1]
    model = RegressionModel(input_dim)
    optimizer = optim.AdamW(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=1e-6, last_epoch=-1)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    best_mse = 100
    for epoch in range(18):
        scheduler.step()
        mean_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

        # 5. 测试模型
        model.eval()
        with torch.no_grad():
            train_predictions = model(torch.tensor(X_train, dtype=torch.float32)).squeeze()
            train_mse = mean_squared_error(y_train, train_predictions.numpy())
            test_predictions = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
            mse = mean_squared_error(y_test, test_predictions.numpy())
            
        if mse < best_mse:
            torch.save(model.state_dict(), f'regression_model_{seed}.pth')
            best_mse = mse
        print(f'Epoch: {epoch}, Train MSE:{train_mse}, Test MSE: {mse}, lr : {scheduler.get_last_lr()[0]}')
    mses.append(best_mse)
print('------------------------------------BEST MSE:',mses,np.mean(mses))
models =[]
# 6. 保存模型
for seed in range(1,18):
    input_dim = X_train.shape[1]
    model = RegressionModel(input_dim)
    model.load_state_dict(torch.load(f'regression_model_{seed}.pth'))
    model.eval()
    models.append(model)
Xs =[]
for model_name in model_names:
    F = pd.read_parquet(model_name.replace('/','.') + '_val.parquet')
    if 'text_feature' in F.keys():
        X = np.array([np.concatenate([F.iloc[i].image_feature[0],F.iloc[i].text_feature[0]]) for i in tqdm(range(len(F)))])
    else:
        X = np.array([F.iloc[i].image_feature for i in tqdm(range(len(F)))])
    X = X/np.linalg.norm(X)
        
    Xs.append(X)
val_Xs = np.concatenate(Xs,1)
#val_X =  np.array([F.iloc[i].image_feature[0]  for i in range(len(F))])
val_X_scaled = scaler.transform(val_Xs)

with torch.no_grad():
    test_predictions = 0
    for model in models:
        test_predictions += model(torch.tensor(val_X_scaled, dtype=torch.float32)).squeeze().numpy()
F['mos'] = test_predictions/3.
F[['name','mos']].sort_values('name').to_csv('output.txt',index=False,header=False)



