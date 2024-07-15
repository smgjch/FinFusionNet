from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
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

warnings.filterwarnings('ignore')



def calculate_ic(predictions, actuals):
    # Convert inputs to tensors if they aren't already
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(actuals, torch.Tensor):
        actuals = torch.tensor(actuals)

    # Mean centering
    pred_mean = torch.mean(predictions)
    act_mean = torch.mean(actuals)
    centered_predictions = predictions - pred_mean
    centered_actuals = actuals - act_mean

    # Covariance calculation
    covariance = torch.mean(centered_predictions * centered_actuals)

    # Standard deviation calculation
    pred_std = torch.std(centered_predictions)
    act_std = torch.std(centered_actuals)

    # Information Coefficient calculation
    ic = covariance / (pred_std * act_std)
    print(f"pred_mean {pred_mean},act_mean {act_mean},covariance {covariance},pred_std {pred_std},act_std {act_std}")
    # print(ic.item())
    return ic.item()


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):

        model = self.model_dict[self.args.model].Model(self.args).float()
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
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = torch.tensor([])
        labels = torch.tensor([])

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # print(f"Get from data loader, batch_x: \n{batch_x} \n\n batch_y{batch_y}")
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # if batch_y[:, -self.args.pred_len:, -1:].std() == 0:
                #     print(f"Get from data loader, batch_x: \n{batch_x} \n\n batch_y{batch_y}")

                # decoder input
                # print(f"batch x \n {batch_x.shape} \n")

                # print(f"batch y \n {batch_y.shape} \n")
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # # print(f"-------------dec_inp ------------\n{dec_inp.shape}")
                dec_inp = 0

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0 #deprecate since label located at the thrid column
                # f_dim = -1 if self.args.features == 'MS' or self.args.features == 'M' else 0
                if self.model.verbose:
                    print(f"in val output shape {outputs.shape}")

                outputs = outputs[:, -self.args.label_len:, 0:1]
                batch_y = batch_y[:, -self.args.label_len:, 0:1].to(self.device)
                if self.model.verbose:
                    print(f"in val output shape after slice {outputs.shape}")
                    print(f"in val batch_y shape after slice {batch_y.shape}")

                # print(f"------- Label sliced ------- \n{batch_x} ")

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                # print(f"-------pred-----\n{pred.shape},{pred}")
                # print(f"-------true-----\n{true.shape}, {true}")

                preds = torch.cat((preds, pred.reshape(-1)[:-1]), dim=0)
                labels = torch.cat((labels, true.reshape(-1)[:-1]), dim=0)
                # print(f"pre {pred.reshape(-1)[:-1]}\nlabel {true.reshape(-1)[:-1]}")

                loss = criterion(pred, true)
                # ic = calculate_ic(pred, true)

                # ic = np.corrcoef(pred.reshape(-1),true.reshape(-1))

                # if np.isnan(ic).any():
                #     print(f"np ic {ic} \n\n -------pred---------\n {pred} \n\n --------------true-------------\n {true}")
                #     calculate_ic(pred, true)

                total_loss.append(loss)
                # total_ic.append(ic[0][1])


        total_loss = np.average(total_loss)
        ic = np.corrcoef(preds,labels)
        MSE = criterion(preds,labels)
        # total_ic = np.average(total_ic)
        # print(f" {ic[0][1]} \n ---------------- IC ----------------- \n {ic} \n ------------MSE-------------\n{MSE}")
        print(f" {ic[0][1]}")
        
        self.model.train()
        return total_loss

    def train(self, setting):
        print(f"------verbose------- \n {self.args.verbose}")
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
                # print(f"get from loader {batch_y.shape} {batch_y}")
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # print(f"before dec input checkpoint! shapes:  x shape: {batch_x.shape} y shape {batch_y.shape} \n")
                # print(f"before decinput checkpoint! shapes:  x shape: {batch_x.shape} y shape {batch_y.shape} \n x: {batch_x}, \n y: \n {batch_y}")

                # decoder input
                # print(f"batch y \n {batch_y.shape} \n content: \n{batch_y}")

                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # # print(f"checkpoint! shapes:  x shape: {batch_x.shape} y shape {dec_inp.shape} \n x: {batch_x}, \n y: \n {dec_inp}")
                # print(f"checkpoint! shapes:  x shape: {batch_x.shape} y shape {dec_inp.shape} ")

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        # f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.label_len:, 0:1]
                        batch_y = batch_y[:, -self.args.label_len:, 0:1].to(self.device)  # only for btc dataset since target is at first column
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # print("in else")
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, 0, 0)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, 0, 0)
                    
                    if self.model.verbose:
                        print(f"in train, ouput shape before slice {outputs.shape}")
                    
                    f_dim = -1 if self.args.features == 'MS' else 0 # deprecate on btc dataset

                    outputs = outputs[:, -self.args.label_len:, 0:1]
                    

                    if self.model.verbose:    
                        print(f"in train, ouput after slice {outputs.shape}")

                    batch_y = batch_y[:, -self.args.label_len:, 0:1].to(self.device) # batch_y slice label only
                    # print(f"in train, ouput after slice {batch_y.shape} \n {batch_y}")
                    
                    
                    if self.model.verbose:
                        print(f"in train, batch_y shape after slice {batch_y.shape}")
                    
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("val ic: ",end = "")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("test ic: ",end = "")
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

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
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.label_len:, :]
                batch_y = batch_y[:, -self.args.label_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, 0:1]
                batch_y = batch_y[:, :, 0:1]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

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

