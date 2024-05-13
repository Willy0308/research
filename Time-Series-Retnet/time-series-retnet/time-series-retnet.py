import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from multiscale_retention import MultiScaleRetention
from multiscale_retention import theta_shift
import matplotlib.pyplot as plt
import time
import numpy as np
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Check if CUDA is available, then select the GPU by index (e.g., 0, 1, etc.)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

forecast_horizon = 120
look_back = 720

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, look_back=look_back, forecast_horizon=forecast_horizon, feature_columns=None, target_column='Load'):
        self.look_back = look_back
        self.forecast_horizon = forecast_horizon
        self.original_features = dataframe[feature_columns].values if feature_columns else dataframe.drop(columns=[target_column]).values
        self.original_targets = dataframe[[target_column]].values
        
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        self.features = self.feature_scaler.fit_transform(self.original_features)
        self.targets = self.target_scaler.fit_transform(self.original_targets.reshape(-1, 1)).flatten()
        
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features) - self.look_back - self.forecast_horizon + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.look_back]
        y = self.targets[idx + self.look_back:idx + self.look_back + self.forecast_horizon]
        return x, y.view(self.forecast_horizon)
    
    def get_original_data(self, idx):
        """Get untransformed data for visual inspection or other uses."""
        x = self.original_features[idx:idx + self.look_back]
        y = self.original_targets[idx + self.look_back:idx + self.look_back + self.forecast_horizon].flatten()
        return x, y
    
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    best_model_state = None
    
    total_duration = 0  # Initialize total duration

    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing the epoch
        model.train()
        train_loss = []

        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()

        avg_train_loss = sum(train_loss) / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pth')

        end_time = time.time()  # End timing the epoch
        epoch_duration = end_time - start_time
        total_duration += epoch_duration  # Accumulate total duration

        print(f'Epoch [{epoch+1}/{num_epochs}]: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Duration: {epoch_duration:.2f} seconds')

    average_duration = total_duration / num_epochs  # Compute average duration
    print(f'Average Duration per Epoch: {average_duration:.2f} seconds')

    return best_model_state

# Future prediction function
def predict_future(model, dataset, device, forecast_horizon, look_back):

    model.eval()
    # Get the last 'look_back' sequences from the dataset
    if len(dataset.features) >= look_back:
        last_sequence = dataset.features[-look_back:]  # Shape: [look_back, num_features]
    else:
        raise ValueError("Dataset has fewer entries than required for look_back")
    
    # Ensure last_sequence is three-dimensional for model input
    if last_sequence.ndim == 2:
        last_sequence = last_sequence.unsqueeze(0)  # Add a batch dimension: [1, look_back, num_features]
    
    last_sequence = last_sequence.to(device)
    
    with torch.no_grad():
        future_predictions = model(last_sequence).view(-1)  # Ensure it is flattened if needed

    # Scale predictions back to the original scale
    future_predictions = dataset.target_scaler.inverse_transform(future_predictions.cpu().numpy().reshape(-1, 1)).flatten()

    
    return future_predictions


def model_summary(model):
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 计算模型大小
    model_size = total_params * 4 / (1024 ** 2)  # 以浮点数(float32)计算，每个数字4字节
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    print(f"Approximate model size: {model_size:.2f} MB")


# Original dataset
dataframe = pd.read_csv('../ausdata.csv')
feature_columns = ['DryBulb', 'DewPnt', 'WetBulb', 'Humidity', 'ElecPrice', 'Load']
train_size = int(0.7 * len(dataframe))
validate_size = int(0.1 * len(dataframe))
test_size = len(dataframe) - train_size - validate_size
train_data = dataframe.iloc[:train_size]
validate_data = dataframe.iloc[train_size:train_size + validate_size]
test_data = dataframe.iloc[train_size + validate_size:]
# print("test_data len:", len(test_data))

train_dataset = TimeSeriesDataset(train_data, feature_columns=feature_columns, target_column='Load')
validate_dataset = TimeSeriesDataset(validate_data, feature_columns=feature_columns, target_column='Load')
test_dataset = TimeSeriesDataset(test_data, feature_columns=feature_columns, target_column='Load')

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(validate_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Modified dataset
dataframe_mod = pd.read_csv('../modified_ausdata.csv')
train_data_mod = dataframe_mod.iloc[:train_size]
validate_data_mod = dataframe_mod.iloc[train_size:train_size + validate_size]
test_data_mod = dataframe_mod.iloc[train_size + validate_size:]

test_dataset_mod = TimeSeriesDataset(test_data_mod, feature_columns=feature_columns, target_column='Load')

test_loader_mod = DataLoader(test_dataset_mod, batch_size=128, shuffle=False)


################################################################ Train model ############################################################
model = MultiScaleRetention(input_dim=6, value_dim=128, num_heads=32, gate_fn="gelu").to(device)
model_summary(model)
# train_model(model, train_loader, val_loader)

################################################################# Load model #############################################################
best_model_state = torch.load('best_model_720_120_128_100.pth')
model.load_state_dict(best_model_state)

############################################################# Predict future values ######################################################
# Generate predictions using the last sequence from the test dataset
start_time = time.time()  # 開始計時
future_predictions = predict_future(model, test_dataset, device, forecast_horizon, look_back)
end_time = time.time()  # 結束計時
duration = end_time - start_time  # 計算運行時間

print(f'predict_future function duration: {duration:.4f} seconds')

# Number of test data points to display
num_test_points = look_back

# Calculate start index for the last 720 data points in the test dataset
start_index = len(test_dataset.targets) - num_test_points

# Convert these last 720 data points back to their original scale
last_test_values = dataframe['Load'].iloc[len(dataframe)-look_back:len(dataframe)]

# Generate timestamps for these points
start_date_of_last_data = pd.to_datetime(dataframe['date']).iloc[train_size + validate_size + start_index]
test_timestamps = pd.date_range(start=start_date_of_last_data, periods=num_test_points, freq='H')

# Generate future prediction timestamps starting from the last timestamp in test data
future_timestamps = pd.date_range(start=test_timestamps[-1] + pd.Timedelta(hours=1), periods=forecast_horizon, freq='H')

# Concatenate test timestamps with future prediction timestamps
all_timestamps = test_timestamps.union(future_timestamps)

# Combine the last part of test data values with the future predictions
all_load_values = np.concatenate([last_test_values, future_predictions])

# Plotting the last part of test load values and future predictions
plt.figure(figsize=(15, 7))
plt.plot(all_timestamps[:num_test_points], last_test_values, label='Last Part of Test Dataset Load', color='blue')
plt.plot(all_timestamps[num_test_points:], future_predictions, label='Future Predictions', linestyle='--', color='red')
plt.title('Load Forecast Including Last Part of Test Data and Future Predictions')
plt.xlabel('Date')
plt.ylabel('Load')
plt.legend()
plt.grid(False)
plt.savefig('./future_predict.png')
# plt.show()

print("Future Predictions:", future_predictions)

# ----------------------------------------------------------------------------------------
def calculate_metrics(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_mae = 0
    total_mse = 0
    count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            # Calculate MAE
            mae = torch.abs(output - target).mean()
            total_mae += mae.item() * data.size(0)
            
            # Calculate MSE
            mse = ((output - target) ** 2).mean()
            total_mse += mse.item() * data.size(0)
            
            count += data.size(0)

    average_mae = total_mae / count
    average_mse = total_mse / count
    return average_mae, average_mse

# Testing and plotting results
average_mae, average_mse = calculate_metrics(model, test_loader, device)
print("Average MAE on the test set:", average_mae)
print("Average MSE on the test set:", average_mse)

# ----------------------------------------------------------------------------------------
def plot_forecast(model, test_loader, scaler, sample_index=(0, 0), save_path=None):
    model.eval()  # Set the model to evaluation mode
    
    # Navigate to the correct batch
    for batch_index, (data, target) in enumerate(test_loader):
        if batch_index == sample_index[0]:
            break
    
    # Extract the specific sample
    data_sample = data[sample_index[1]].to(device).unsqueeze(0)  # Add batch dimension for single sample
    target_sample = target[sample_index[1]].to(device).unsqueeze(0)

    # Calculate the start index for timestamps based on the DataLoader configuration
    plot_start_index = train_size + validate_size + look_back + sample_index[0] * data.size(0) + sample_index[1]
    timestamps = dataframe['date'].iloc[plot_start_index:plot_start_index + forecast_horizon].reset_index(drop=True)

    with torch.no_grad():
        output = model(data_sample)
        predictions = output.squeeze(0)  # Remove batch dimension for display

    target_actuals = scaler.inverse_transform(target_sample.cpu().numpy().reshape(-1, 1)).flatten()
    predictions_actuals = scaler.inverse_transform(predictions.cpu().numpy().reshape(-1, 1)).flatten()
    mae = torch.mean(torch.abs(predictions - target_sample.squeeze(0)))
    mse = torch.mean((predictions - target_sample.squeeze(0)) ** 2)

    num_points = 5
    indices = np.linspace(0, len(timestamps) - 1, num_points, dtype=int)
    selected_timestamps = [timestamps[i] for i in indices]

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, target_actuals, label='Actual Load', marker='o')
    plt.xticks([timestamps[i] for i in indices], selected_timestamps, rotation=0)
    plt.plot(predictions_actuals, label='Predicted Load', linestyle='--', marker='x')
    plt.title('Load Forecasting Comparison Result')
    plt.xlabel('Date')
    plt.ylabel('Load')
    plt.legend()
    plt.savefig(save_path)
    # plt.show()

    return predictions_actuals, target_actuals, mae.item(), mse.item()

# Testing and plotting results
predictions_actuals, target_actuals, mae, mse = plot_forecast(
    model, test_loader, test_dataset.target_scaler, 
    sample_index=(10, 0),  # Change to any (batch_index, sample_index) you want
    save_path='./forecast_comparison.png'
)
# print("Predictions:", predictions_actuals)
# print("Actuals:", target_actuals)
# print("MAE:", mae)
# print("MSE:", mse)

# ------------------------------------------------------------------------------------------
# Anomaly detection setup
def anomaly_detection(test_loader, model, scaler, forecast_horizon):
    model.eval()
    total_time_points = len(test_loader.dataset)  # 这应该已经正确
    time_point_scores = [[] for _ in range(total_time_points + forecast_horizon - 1)]  # 调整长度

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = scaler.inverse_transform(output.cpu().numpy())
            actual = scaler.inverse_transform(target.cpu().numpy())

            if predicted.shape != actual.shape:
                predicted = np.reshape(predicted, actual.shape)

            # Compute anomaly score for each prediction within the forecast window
            for i in range(data.size(0)):  # Loop over each sample in the batch
                for j in range(forecast_horizon):  # Loop over each forecast step
                    index = idx * data.size(0) + i + j  # Global index of the actual data point
                    if index < len(time_point_scores):
                        error = np.abs(predicted[i, j] - actual[i, j])
                        score = error / np.mean(np.abs(predicted[i] - actual[i]))
                        time_point_scores[index].append(score)

    return time_point_scores

def finalize_anomalies(time_point_scores, threshold=0.4):
    anomalies = []
    for scores in time_point_scores:
        # All scores must exceed the threshold for the point to be considered anomalous
        if all(score > threshold for score in scores):
            anomalies.append(True)
        else:
            anomalies.append(False)
    return anomalies

###################################################################################################################################
#--------------------------------------------------------------Original-----------------------------------------------------------#
###################################################################################################################################

# Assuming forecast_horizon + 720 covers your 721st point to the end
time_point_scores = anomaly_detection(test_loader, model, test_dataset.target_scaler, forecast_horizon)
time_point_scores_shap = time_point_scores
anomalies = finalize_anomalies(time_point_scores, threshold=0.4)
time_point_scores = time_point_scores[29:7301]
# print("time_point_scores len:", len(time_point_scores))
anomalies = finalize_anomalies(time_point_scores, threshold=0.4)
# print("anomalies len:", len(anomalies))
# Assuming you have loaded your dataframe and set up your dataset properly
# Adjust the indices for plotting by considering look_back and forecast_horizon
plot_start_index = train_size + validate_size + look_back  # This is where the test data effectively starts in the original dataframe
plot_end_index = train_size + validate_size + len(test_data)  # This is the end of the test data in the original dataframe

# Get the timestamps for the adjusted test data range
timestamps = dataframe['date'].iloc[plot_start_index:plot_end_index].reset_index(drop=True)
timestamps = timestamps[29:7301].reset_index(drop=True)

load_values = test_dataset.targets[:(plot_end_index - plot_start_index)]
load_values = load_values[29:7301]

# Fetch the unscaled original load values directly for the range being plotted
unscaled_load_values = test_dataset.original_targets[look_back:].flatten()
unscaled_load_values = unscaled_load_values[29:7301]

# Anomalies need to be rechecked to ensure they only include indices within the valid plotting range
valid_anomaly_indices = [i for i in range(len(anomalies)) if anomalies[i] and (i < len(load_values))]

# 获取正常样本的索引
normal_indices = [i for i in range(len(load_values)) if i not in valid_anomaly_indices]

# Extract the actual load values for anomalies
anomaly_load_values = [unscaled_load_values[i] for i in valid_anomaly_indices]

plt.figure(figsize=(15, 5))
plt.plot(timestamps, unscaled_load_values, label='Original Load Values', color='blue')
plt.scatter([timestamps[i] for i in valid_anomaly_indices], anomaly_load_values, color='red', s=25, label='Anomalies Detected', zorder=2)
plt.title('Anomaly Detection for Original Dataset')
plt.xlabel('Date')
plt.ylabel('Load Value')
plt.legend()

# Calculate the total number of data points and the percentage of anomalies
total_points = len(load_values)
num_anomalies = len(valid_anomaly_indices)
percentage_anomalies = (num_anomalies / total_points) * 100

# Adding text annotation for anomalies count and percentage
plt.text(0.01, 0.95, f'Anomalies detected: {num_anomalies} ({percentage_anomalies:.2f}%)', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Customize the x-ticks to show only selected dates for clarity
num_points = 5
indices = np.linspace(0, len(timestamps) - 1, num_points, dtype=int)
selected_timestamps = [timestamps[i] for i in indices]
plt.xticks([timestamps[i] for i in indices], selected_timestamps, rotation=0)

plt.tight_layout()
plt.grid(True)
plt.savefig('anomaly_original.png')
# plt.show()

###################################################################################################################################
#--------------------------------------------------------------Modified-----------------------------------------------------------#
###################################################################################################################################
def convert_to_integer_labels(anomalies):
    """Convert boolean anomaly detection results to integer labels."""
    return [int(anomaly) for anomaly in anomalies]

def evaluate_thresholds(time_point_scores, labels_actual, thresholds):
    results = []
    for threshold in thresholds:
        anomalies_boolean = finalize_anomalies(time_point_scores, threshold=threshold)
        anomalies_integer = convert_to_integer_labels(anomalies_boolean)

        # Calculate metrics
        accuracy = accuracy_score(labels_actual, anomalies_integer)
        precision = precision_score(labels_actual, anomalies_integer)
        recall = recall_score(labels_actual, anomalies_integer)
        f1 = f1_score(labels_actual, anomalies_integer)

        # Store results
        results.append({
            'Threshold': threshold,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    return results

# Assuming forecast_horizon + 720 covers your 721st point to the end
load_values = test_dataset_mod.targets[:(plot_end_index - plot_start_index)]
load_values = load_values[29:7301]

# Fetch the unscaled original load values directly for the range being plotted
unscaled_load_values = test_dataset_mod.original_targets[look_back:].flatten()
unscaled_load_values = unscaled_load_values[29:7301]

# Load the labels from the dataset for the corresponding test data
labels_actual = test_data_mod.iloc[look_back:]['Anomaly'].values
labels_actual = labels_actual[29:7301]

# Fetch anomaly detection results
time_point_scores = anomaly_detection(test_loader_mod, model, test_dataset_mod.target_scaler, forecast_horizon)
time_point_scores = time_point_scores[29:7301]

# # Evaluate thresholds
# thresholds = np.arange(0.1, 1.1, 0.1)  # 0.1 to 1.0
# results = evaluate_thresholds(time_point_scores, labels_actual, thresholds)

# # Print the results
# for result in results:
#     print(f"Threshold: {result['Threshold']:.1f}, "
#           f"Accuracy: {result['Accuracy']:.3f}, "
#           f"Precision: {result['Precision']:.3f}, "
#           f"Recall: {result['Recall']:.3f}, "
#           f"F1 Score: {result['F1 Score']:.3f}")


# Assuming forecast_horizon + 720 covers your 721st point to the end
anomalies = finalize_anomalies(time_point_scores, threshold=0.4)

# Anomalies need to be rechecked to ensure they only include indices within the valid plotting range
valid_anomaly_indices = [i for i in range(len(anomalies)) if anomalies[i] and (i < len(load_values))]

# Extract the actual load values for anomalies
anomaly_load_values = [unscaled_load_values[i] for i in valid_anomaly_indices]

plt.figure(figsize=(15, 5))
plt.plot(timestamps, unscaled_load_values, label='Original Load Values', color='blue')
plt.scatter([timestamps[i] for i in valid_anomaly_indices], anomaly_load_values, color='red', s=25, label='Anomalies Detected', zorder=2)
plt.title('Anomaly Detection for Testing')
plt.xlabel('Date')
plt.ylabel('Load Value')
plt.legend()


# Adding text annotation for anomalies count and percentage
plt.text(0.01, 0.95, f'Anomalies detected: {num_anomalies} ({percentage_anomalies:.2f}%)', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.xticks([timestamps[i] for i in indices], selected_timestamps, rotation=0)
plt.tight_layout()
plt.grid(True)
plt.savefig('anomaly_test.png')
# plt.show()

# print("valid_anomaly_indices length:", len(valid_anomaly_indices))
# print("valid_anomaly_indices:", valid_anomaly_indices)
# Assuming the setup and model loading as previously described

# 检查模型的输入形状
sample_input = torch.zeros((1, look_back, len(feature_columns)), dtype=torch.float32).to(device)
print("Sample Input Shape:", sample_input.shape)

# 使用模型进行预测
with torch.no_grad():
    model.eval()
    sample_output = model(sample_input)
    print("Sample Output Shape:", sample_output.shape)


# Step 1: Identify an anomalous time point
# For this example, let's assume 'valid_anomaly_indices[0]' gives us our first detected anomaly index
anomalous_index = valid_anomaly_indices[100]

# Extract the corresponding data
anomalous_data = test_dataset.features[anomalous_index - look_back:anomalous_index].unsqueeze(0).to(device)

print("anomalous_data shape:", anomalous_data.shape)

# Step 2: Prepare background data
def prepare_background_data(dataset, valid_anomaly_indices, num_samples=100):
    # 得到所有可能的索引，去除异常索引
    possible_indices = [i for i in range(len(dataset) - look_back) if i not in valid_anomaly_indices]
    
    # 从剩余的正常索引中随机选择JO4G
    selected_indices = np.random.choice(possible_indices, size=min(num_samples, len(possible_indices)), replace=False)
    
    # 从选中的正常索引中获取数据
    background_samples = [dataset.features[idx:idx + look_back].unsqueeze(0) for idx in selected_indices]
    
    # 合并成一个批量数据
    background_data = torch.cat(background_samples, dim=0).to(device)
    return background_data

# 调用函数准备 background_data
background_data = prepare_background_data(test_dataset, valid_anomaly_indices)

# print("background_data shape:", background_data.shape)

explainer = shap.DeepExplainer(model, background_data)

# Step 4: Compute SHAP values for the anomalous data

shap_values = explainer.shap_values(anomalous_data, check_additivity=False)

# for i, shap_value in enumerate(shap_values):
#     print(f"SHAP values for output {i}: {np.array(shap_value).shape}")

# print("shap_values:", shap_values)

# Step 4: Visualize SHAP values

shap.initjs()
force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0][0][719], feature_names=feature_columns)

# Save the plot to an HTML file
shap.save_html('shap_force_plot_anomalous.html', force_plot)







# # 假设你有一个异常样本索引的列表，名为 valid_anomaly_indices
# if len(valid_anomaly_indices) > 0:

#     # 确保单个样本同样拥有正确的维度
#     sample_to_explain = test_dataset.features[sample_index].unsqueeze(0).unsqueeze(0)  # 也添加序列长度维度

#     shap_values = explainer.shap_values(sample_to_explain)
#     expected_value = explainer.expected_value

#     if isinstance(shap_values, list):
#         shap_values = [s.numpy() if hasattr(s, 'numpy') else s for s in shap_values]
#     if hasattr(expected_value, 'numpy'):
#         expected_value = expected_value.numpy()

#     # 初始化 JS 可视化环境
#     shap.initjs()

#     # 生成力量图（Force Plot）
#     force_plot = shap.force_plot(
#         expected_value, 
#         shap_values[0],  # 注意这里可能需要根据您的数据结构进行适当的调整
#         sample_to_explain.squeeze().numpy(),  # 使用实际的特征值，确保没有额外的维度
#         feature_names=feature_columns  # 确保这个属性存在或直接提供列名列表
#     )

#     # 保存力量图为 HTML 文件
#     shap.save_html("./force_plot.html", force_plot)


# def anomaly_detection_collect_data(test_loader, model, scaler, forecast_horizon, device):
#     model.eval()
#     total_time_points = len(test_loader.dataset)
#     time_point_scores = [[] for _ in range(total_time_points + forecast_horizon - 1)]
#     collected_data = []  # To store data for SHAP analysis
#     total_batches = len(test_loader)
#     batch_count = 0
#     start_time = time.time()

#     with torch.no_grad():
#         for idx, (data, target) in enumerate(test_loader):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             predicted = scaler.inverse_transform(output.cpu().numpy())
#             actual = scaler.inverse_transform(target.cpu().numpy())

#             if predicted.shape != actual.shape:
#                 predicted = np.reshape(predicted, actual.shape)

#             for i in range(data.size(0)):  # Loop over each sample in the batch
#                 for j in range(forecast_horizon):  # Loop over each forecast step
#                     index = idx * data.size(0) + i + j  # Global index of the actual data point
#                     if index < len(time_point_scores):
#                         error = np.abs(predicted[i, j] - actual[i, j])
#                         score = error / np.mean(np.abs(predicted[i] - actual[i]))
#                         time_point_scores[index].append(score)
#                         if score > 0.4:  # Threshold for collecting SHAP data
#                             collected_data.append((data[i], target[i], output[i]))
            
#             # Progress tracking
#             batch_count += 1
#             elapsed_time = time.time() - start_time
#             if batch_count % 10 == 0 or batch_count == total_batches:  # Update every 10 batches or on the last batch
#                 percent_complete = (batch_count / total_batches) * 100
#                 estimated_total_time = (elapsed_time / batch_count) * total_batches
#                 estimated_remaining_time = estimated_total_time - elapsed_time
#                 print(f"Progress: {percent_complete:.2f}% complete, Estimated time remaining: {estimated_remaining_time:.2f} seconds")

#     return time_point_scores, collected_data


# # Function to prepare data for SHAP
# def prepare_data_for_shap(test_loader, device):
#     # Get a batch of data
#     for data, target in test_loader:
#         return data.to(device), target.to(device)

# # Function to apply SHAP Deep Explainer
# def apply_shap_deep_explainer(model, data):
#     # Initialize the background data
#     background_data = data[:1]  # Use the first 100 samples for simplicity; adjust as needed
    
#     # Create the explainer
#     explainer = shap.DeepExplainer(model, background_data)
    
#     # Generate SHAP values for the same set of data
#     shap_values = explainer.shap_values(data)
    
#     # Plot SHAP values for the first prediction
#     shap.initjs()
#     shap.force_plot(explainer.expected_value[0], shap_values[0][0], data[0].cpu().numpy())

# # Run the anomaly detection to collect data
# time_point_scores, collected_data = anomaly_detection_collect_data(test_loader, model, test_dataset.target_scaler, forecast_horizon, device)

# print(collected_data )
# # Assuming collected_data is a list of tuples (input, target, output)
# # Extract just the inputs for SHAP
# data_for_shap = torch.stack([x[0] for x in collected_data]).to(device)

# print("fuck2")
# # Apply SHAP Deep Explainer
# apply_shap_deep_explainer(model, data_for_shap)











