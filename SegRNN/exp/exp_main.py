from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, VanillaRNN, SegRNN, Retnet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from IPython.display import Image, display

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

# Add lime import
from lime import lime_tabular

import shap

warnings.filterwarnings('ignore')

# class ModelOutputWrapper(nn.Module):
#     def __init__(self, model, output_index=0):
#         super(ModelOutputWrapper, self).__init__()
#         self.model = model
#         self.output_index = output_index

#     def forward(self, x):
#         self.model.eval()
#         # Assuming self.model.rnn is your RNN layer. Adjust based on your model.
#         self.model.rnn.train()
#         x.requires_grad_(True)
#         outputs = self.model(x)
#         selected_output = outputs[:, self.output_index]
#         return selected_output.unsqueeze(-1)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.model = self._build_model()  # Build the model
        self.model.to(self.device)  # Ensure model is on the correct device
        self._print_model_size()  # Print the model size after building
        # self.model = self._build_model()
        # # 注意：加载模型状态字典，确保模型被正确初始化
        # self.model = torch.load('/home/willy/research/apartment/checkpoints/Electricity_96_96_SegRNN_custom_ftM_sl96_pl96_dm512_dr0.1_rtgru_dwpmf_sl48_test_0_ausdata/complete_model.pt')
        # self.model.to(self.device)
        # self.model.eval()

    def lime_explain(self, data_loader, instance_index=0):
        self.model.eval()
        # Adjust the unpacking here to match what the DataLoader yields
        background_batch = next(iter(data_loader))[0]  # Only take the first item, assuming it's the data
        background_data = background_batch.numpy()  # Convert to NumPy array for LIME
        
        # Flatten the time series data for LIME
        num_samples, sequence_length, num_features = background_data.shape
        background_data_flat = background_data.reshape(num_samples, sequence_length * num_features)
        
        # Define feature names for the flattened data
        feature_names = [f'Time_{i}_{j}' for i in range(sequence_length) for j in ['DryBulb', 'DewPnt', 'WetBulb', 'Humidity', 'ElecPrice', 'OT']]
        
        # Initialize LIME Explainer for flattened data
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=background_data_flat,
            feature_names=feature_names,
            class_names=['OT'],
            mode='regression'
        )
        
        # Select a specific instance to explain and flatten it
        instance = background_data_flat[instance_index]

        # Define a prediction function for flattened data
        def predict_fn(x):
            self.model.eval()
            with torch.no_grad():
                # Assuming x has shape (num_samples, sequence_length * num_features) for LIME
                # First, reshape x back to the expected input shape for the model
                x_reshaped = x.reshape(-1, sequence_length, num_features)
                input_tensor = torch.from_numpy(x_reshaped).to(dtype=torch.float32, device=self.device)
                
                # Get model predictions
                predictions = self.model(input_tensor)
                
                # Select predictions for the last timestep and a specific feature (e.g., 'OT')
                # Adjust the indexing [-1, -1] if 'OT' is not the last feature
                selected_predictions = predictions[:, -1, -1]  # Last timestep, 'OT' feature
                
                # Ensure we return a 1D array: one prediction per sample
                return selected_predictions.cpu().numpy()
        


        # Number of features you want to display in the explanation
        num_features_to_display = 10  # Adjust this to display more or fewer features

        # Generate LIME explanation for the selected instance
        # Specify the num_features parameter to limit the number of features in the explanation
        exp = explainer.explain_instance(
            instance, 
            predict_fn, 
            num_features=num_features_to_display  # This will limit the number of features
        )

        # Create a larger figure to accommodate the features
        plt.figure(figsize=(10, 10))  # Adjust the size as needed

        # Generate the explanation plot
        exp.as_pyplot_figure()
        plt.title(f'Explanation for instance {instance_index}')
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

        # Automatically adjust subplot params so that the subplot(s) fits into the figure area
        plt.tight_layout()

        # Save the figure with a higher DPI for better resolution
        figures_dir = '/home/willy/research/apartment/'
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        fig_path = os.path.join(figures_dir, f'lime_explanation_instance_{instance_index}.png')
        plt.savefig(fig_path, dpi=300)

        # Close the plot to free up memory
        plt.close()




    # def check_model_output(self, data_loader):
    #     self.model.eval()
    #     batch_x, _, _, _ = next(iter(data_loader))
    #     batch_x = batch_x.float().to(self.device)
    #     with torch.no_grad():
    #         outputs = self.model(batch_x)
    #     print("Output shape:", outputs.shape)

        
        

    # In Exp_Main or wherever the SHAP explanation is being initiated
    
    # def shap_explain(self, data_loader, output_index=0):
    #     self.model.eval()  # Ensure the model is in evaluation mode.

    #     # Select a background batch for SHAP explainer.
    #     background_batch = next(iter(data_loader))
    #     background_data = background_batch[0].float().to(self.device)

    #     # Wrap the model for SHAP.
    #     model_wrapper = ModelOutputWrapper(self.model, output_index=output_index).to(self.device)

    #     # Initialize SHAP DeepExplainer.
    #     explainer = shap.DeepExplainer(model_wrapper, background_data)

    #     # Select a test batch to explain.
    #     test_batch = next(iter(data_loader))
    #     test_data = test_batch[0].float().to(self.device)

    #     # Compute SHAP values. Note: 'nsamples' parameter is not required for DeepExplainer.
    #     try:
    #         shap_values = explainer.shap_values(test_data)
    #     except Exception as e:
    #         print(f"An error occurred when computing SHAP values: {e}")
    #         return
            
    #     # Flatten or average the SHAP values across the sequence length.
    #     shap_values_reshaped = shap_values[output_index].sum(axis=1)

    #     print(shap_values_reshaped)

    #     feature_names = ['DryBulb', 'DewPnt', 'WetBulb', 'Humidity', 'ElecPrice', 'OT']

    #     # Compute mean absolute SHAP values.
    #     mean_shap_values = np.abs(shap_values_reshaped).mean(axis=0)

    #     # Compute relative importance.
    #     relative_importance = mean_shap_values / mean_shap_values.sum()

    #     # Flatten or average the test data across the sequence length.
    #     test_data_reshaped = test_data.detach().cpu().numpy().mean(axis=1)

    #     # Use the reshaped data for visualization.
    #     try:
    #         shap.summary_plot(shap_values_reshaped, test_data_reshaped, feature_names=feature_names)
    #     except Exception as e:
    #         print(f"An error occurred when plotting SHAP values: {e}")
    #         return  # Exit the function if plotting cannot be performed.

    #     # 如果需要，也可以保存圖片
    #     plt.savefig('/home/willy/research/apartment/shap_summary_plot.png')
    #     # 绘制平均绝对SHAP值条形图
    #     plt.figure(figsize=(10, 8))
    #     plt.barh(feature_names, mean_shap_values, color='blue')
    #     plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
    #     plt.tight_layout()
    #     plt.savefig('/home/willy/research/apartment/mean_shap_plot.png')
    #     plt.show()

    #     # 绘制特征的相对重要性条形图
    #     plt.figure(figsize=(10, 8))
    #     plt.barh(feature_names, relative_importance, color='green')
    #     plt.xlabel('Relative Importance')
    #     plt.tight_layout()
    #     plt.savefig('/home/willy/research/apartment/relative_importance_plot.png')
    #     plt.show()

    # Additional code for handling SHAP values and plotting...


        # shap.summary_plot(shap_values, test_data.detach().cpu().numpy())
        # shap.summary_plot(shap_values, test_data.detach().cpu().numpy(), feature_names=['DryBulb', 'DewPnt', 'WetBulb', 'Humidity', 'ElecPrice'])

    def _print_model_size(self):
        # Calculate and print the model size in MB
        torch.save(self.model.state_dict(), "temp_model.pth")
        model_size = os.path.getsize("temp_model.pth") / (1024 * 1024)  # Convert from bytes to megabytes
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        os.remove("temp_model.pth")  # Clean up the temporary file
        print(f"Model size: {model_size:.2f} MB")
        print(f"Number of trainable parameters: {num_params}")


    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Retnet': Retnet,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'VanillaRNN': VanillaRNN,
            'SegRNN': SegRNN
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()

        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        max_memory_used = 0  # Variable to store the maximum GPU memory usage
        epoch_times = []  # List to store duration of each epoch

        for epoch in range(self.args.train_epochs):
            epoch_start_time = time.time()  # Start time of the epoch

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("outputs:", outputs)
                    # print("batch_y:", batch_y)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                # Update max GPU memory usage
                current_max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to MB
                max_memory_used = max(max_memory_used, current_max_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            epoch_duration = time.time() - epoch_start_time  # Duration of the epoch
            epoch_times.append(epoch_duration)  # Append duration to list

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        average_epoch_duration = sum(epoch_times) / len(epoch_times)
        print(f'Average Epoch Duration: {average_epoch_duration:.2f} seconds')

        # After training, print the maximum memory usage
        print(f"Maximum GPU Memory Used: {max_memory_used:.2f} MB")

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        torch.save(self.model.state_dict(), best_model_path)
        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        begin_time = time.time()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        ms = (time.time() - begin_time) * 1000 / len(test_data)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        
        # print("預測總數:")
        # print(preds.shape[0] * preds.shape[1])
        # print("------------------------------------------------------")
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # preds = np.load('/home/willy/research/SegRNN/results/combined_result_96.npy')
        # print(preds.flatten()[:5])
        # print(trues.flatten()[:5])
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, ms/sample:{}'.format(mse, mae, ms))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, ms/sample:{}'.format(mse, mae, ms))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        start_time = time.time() 
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {'Linear', 'SegRNN', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        end_time = time.time()  # 結束計時
        duration = end_time - start_time  # 計算運行時間
        print(f'Total prediction time: {duration:.4f} seconds')

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
