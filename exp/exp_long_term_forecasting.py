from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils import tensorboard_helper
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from datetime import datetime

warnings.filterwarnings('ignore')

class CustomLoss(nn.Module):
    def __init__(self, lambda_var=0.1):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_var = lambda_var

    def forward(self, predictions, targets):
        # Calculate the MSE loss
        mse = self.mse_loss(predictions, targets)
        
        # Calculate the standard deviation of the predictions
        pred_std = torch.std(predictions, unbiased=False)
        label_std = torch.std(targets, unbiased=False)
        
        # Calculate the variance penalty
        var_penalty = (label_std - pred_std) ** 2
        
        # Combine MSE loss with variance regularization
        loss = mse + self.lambda_var * var_penalty
        # print(f"penalty {var_penalty}")
        return loss

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        now = datetime.now()
        
        time_string = now.strftime("%Y-%m-%d %H:%M:%S")
        writer = SummaryWriter(log_dir=f'train_log/{args.model_id}{time_string}')
        self.loger = tensorboard_helper.Loger(writer = writer, global_step = 0)


    def _build_model(self):

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        amount = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {amount}")
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # criterion = CustomLoss(lambda_var=1)
        # print("CustomLoss activated!")
        return criterion

    # def _select_criterion(self):
    #     # print(f"yes triggered")
    #     def IC_MSE(output, target):
    #         alpha = 0.2
    #         # MSE Loss
    #         mse_loss = nn.functional.mse_loss(output, target)

    #         # Convert tensors to numpy arrays
    #         output_np = output.detach().cpu().numpy().flatten()
    #         target_np = target.detach().cpu().numpy().flatten()
            
    #         # Calculate correlation coefficient using np.corrcoef
    #         corrcoef_matrix = np.corrcoef(output_np, target_np)
    #         correlation_coefficient = corrcoef_matrix[0][1]
    #         # Combined Loss
    #         # print(f"ic {correlation_coefficient}")
    #         combined_loss = alpha * mse_loss + (1 - alpha) * (1 - correlation_coefficient)
    #         # print(f"combined_loss type {type(combined_loss)},mse_loss type {type(mse_loss)},mse_loss item {type(mse_loss.item())}, combined_loss item {type(combined_loss.item())}")
    #         # print(f"mse loss {mse_loss}, conbined loss {combined_loss}")
    #         return combined_loss
    
    #     return IC_MSE

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # print(f"len input shape {len(vali_data)}")
        preds = torch.tensor([])
        labels = torch.tensor([])

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # print(f"VAL Get from data loader \n batch_y{batch_y.shape}")
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs =  self.model(batch_x, 0, 0, 0)[0]
                        else:
                            outputs =  self.model(batch_x, 0, 0, 0)

                else:
                    if self.args.output_attention:
                        outputs =  self.model(batch_x, 0, 0, 0)[0]
                    else:
                        outputs =  self.model(batch_x, 0, 0, 0)
                f_dim = -1 if self.args.features == 'MS' else 0 #deprecate since label located at the thrid column
                # f_dim = -1 if self.args.features == 'MS' or self.args.features == 'M' else 0
                if self.model.verbose:
                    print(f"in val output shape {outputs.shape}")

                outputs = outputs[:, -self.args.label_len:, 0:1]
                batch_y = batch_y[:, -self.args.label_len:, 0:1].to(self.device)
                if self.model.verbose:
                    print(f"in val output shape after slice {outputs.shape}")
                    print(f"in val batch_y shape after slice {batch_y.shape}")

                # print(f"------- Label sliced ------- \n{batch_y.shape} ")

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                # print(f"-------pred-----\n{pred.shape},{pred}")
                # print(f"-------true-----\n{true.shape}, {true}")

                preds = torch.cat((preds, pred.reshape(-1)), dim=0)
                labels = torch.cat((labels, true.reshape(-1)), dim=0)
                # print(f"------- Label cated ------- \n{labels.shape} ")

                mse_loss = nn.functional.mse_loss(pred, true)


                total_loss.append(mse_loss.item())
                # total_ic.append(ic[0][1])


        mean_loss = np.average(total_loss)
        ic = np.corrcoef(preds,labels)
        print(f" {ic[0][1]}\n")
        print(f"-----------prediction------- \n{preds}\n")

        print(f"-----------Lables------- \n{labels}\n")
        print(f"mean of prediction {preds.mean()}, std of prediction {preds.std()}, len {len(preds)}")
        print(f"mean of labels {labels.mean()}, std of labels {labels.std()}, len {len(labels)}")

        
        self.model.train()
        return mean_loss, ic[0][1]
      
    def train(self, setting):

        if self.args.write_graph:
            sample_x, _x, y, _y= tensorboard_helper.get_sample_input(self.args,self.device)
            self.loger.writer.add_graph(self.model, (sample_x, _x, y, _y))
        
        if self.args.log_gradient:
            self.loger.log_gradients(self.model)



        # print(f"------verbose------- \n {self.args.verbose}")
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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # print(f"Train Get from data loader, batch_x: \n{batch_x.shape} \n\n batch_y{batch_y.shape}")
                # print(f"get from loader {batch_x.shape}")
                self.loger.global_step += 1
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
              
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs =  self.model(batch_x, 0, 0, 0)[0] # not entered does not matter
                        else:
                            outputs =  self.model(batch_x, 0, 0, 0)
                            # print(f"tain output shape {outputs.shape}")
                        # f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.label_len:, 0:1]
                        batch_y = batch_y[:, -self.args.label_len:, 0:1].to(self.device) # only for btc dataset since target is at first column
                        # print(f"indeed here")

                        loss = criterion(outputs, batch_y)
                        mse_loss = nn.functional.mse_loss(outputs, batch_y)

                        train_loss.append(mse_loss.item())

                else:
                    # print("in else")
                    if self.args.output_attention:
                        outputs =  self.model(batch_x, 0, 0, 0)[0]
                    else:
                        outputs =  self.model(batch_x, 0, 0, 0)
                    
                    if self.model.verbose:
                        print(f"in train, ouput shape before slice {outputs.shape}")
                    
                    f_dim = -1 if self.args.features == 'MS' else 0 # deprecate on btc dataset


                    outputs = outputs[:, -self.args.label_len:, 0:1]
                    

                    if self.model.verbose:    
                        print(f"in train, ouput after slice {outputs.shape}")

                    batch_y = batch_y[:, -self.args.label_len:, 0:1].to(self.device) 
                    # batch_y slice label only


                    
                    
                    if self.model.verbose:
                        print(f"in train, batch_y shape after slice {batch_y.shape}")
                    
                    loss = criterion(outputs, batch_y)
                    mse_loss = nn.functional.mse_loss(outputs, batch_y)

                    train_loss.append(mse_loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, mse_loss))
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("\nval ic: ",end = "")
            vali_loss, vic = self.vali(vali_data, vali_loader, criterion)
            print("\ntest ic: ",end = "")
            test_loss, tic = self.vali(test_data, test_loader, criterion)

            self.loger.writer.add_scalar('train_loss', train_loss, epoch)
            self.loger.writer.add_scalar('val_loss', vali_loss, epoch)
            self.loger.writer.add_scalar('test_loss', test_loss, epoch)
            self.loger.writer.add_scalar('test ic', tic, epoch)
            self.loger.writer.add_scalar('val ic', vic, epoch)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # convlayers = [self.model.conv1_layers, self.model.conv1_1_layers,
        #               self.model.conv2_layers,self.model.conv2_1_layers,
        #               self.model.conv3_layers,self.model.conv3_1_layers]
        
        # for layer in convlayers:
        #     kernels = layer.weight.data.clone()
        #     num_filters = kernels.size(0)
        #     num_channels = kernels.size(1)
        #     print(kernels)


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # if not self.args.gradient_checkpoint:
        #     for name, param in self.model.named_parameters():
        #         self.loger.writer.add_histogram(name + '_grad', param.grad, global_step=self.loger.global_step)

        self.loger.writer.close()
        

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # # encoder - decoder
                dec_inp = 0
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs =  self.model(batch_x, 0, 0, 0)[0]
                        else:
                            outputs =  self.model(batch_x, 0, 0, 0)
                else:
                    if self.args.output_attention:
                        outputs =  self.model(batch_x, 0, 0, 0)[0]

                    else:
                        outputs =  self.model(batch_x, 0, 0, 0)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.label_len:, :]
                batch_y = batch_y[:, -self.args.label_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()


                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                # print(f"shapes beforo outputs {outputs.shape}, batch_y {batch_y.shape}")

                outputs = outputs[:, :, 0:1]
                batch_y = batch_y[:, :, 0:1]

                
                pred = outputs
                true = batch_y
                
                # print(f"shapes beforo visual pred {pred.shape}, true {true.shape}")

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    # print(f"when testing input {input.shape}, pred {pred.shape}")
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
