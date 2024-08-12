from data_provider.data_factory import data_provider
from data_provider.data_loader import mDataset_btc_CGNN
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

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
import pandas as pd

from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from torch.utils.tensorboard import SummaryWriter


warnings.filterwarnings('ignore')



class Exp_Long_Term_Forecast_CGNN(Exp_Long_Term_Forecast):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_CGNN, self).__init__(args)

    def _select_optimizer(self):
        model_optim = optim.Rprop(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = torch.tensor([])
        labels = torch.tensor([])

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(vali_loader):
                # print(f"Get from data loader, batch_x: \n{batch_x} \n\n batch_y{batch_y}")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                edge_index, edge_attr = self.get_edges_when_training(batch_x,training=False)
                edge_index, edge_attr = edge_index.to(self.device), edge_attr.to(self.device)
                # print(f"edge {edge_index} \n attr {edge_attr}")
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs =  self.model(batch_x, edge_index, edge_attr)[0]
                        else:
                            outputs =  self.model(batch_x,  edge_index, edge_attr)
                else:
                    if self.args.output_attention:
                        outputs =  self.model(batch_x, edge_index, edge_attr)[0]
                    else:
                        outputs =  self.model(batch_x,  edge_index, edge_attr)
                f_dim = -1 if self.args.features == 'MS' else 0 #deprecate since label located at the thrid column
                # f_dim = -1 if self.args.features == 'MS' or self.args.features == 'M' else 0
                if self.model.verbose:
                    print(f"in val output shape {outputs.shape}")

                # print(f"in val output shape {outputs}")

                outputs = outputs[:, -self.args.label_len:, 0:1]
                # print(f"in val output shape after slice {outputs}")

                batch_y = batch_y[:, -self.args.label_len:, 0:1].to(self.device)
                if self.model.verbose:
                    print(f"in val output shape after slice {outputs.shape}")
                    print(f"in val batch_y shape after slice {batch_y.shape}")

                # print(f"------- Label sliced ------- \n{batch_x} ")

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                # if np.isnan(pred).any():
                #     print(f"Get from data loader, batch_x: \n{batch_x} \n\n batch_y{batch_y}\n\n")
                #     print(f"edge_index {edge_index} \nedge_attr {edge_attr}")
                #     print(f"detected nan in pred {pred}")
                # else:
                #     print(f"你过关！！！")

                preds = torch.cat((preds, pred.reshape(-1)[:-1]), dim=0)
                labels = torch.cat((labels, true.reshape(-1)[:-1]), dim=0)
                # print(f"pre {pred.reshape(-1)[:-1]}\nlabel {true.reshape(-1)[:-1]}")

                loss = criterion(pred, true)
                # ic = calculate_ic(pred, true)

                # ic = np.corrcoef(pred.reshape(-1),true.reshape(-1))

                #     print(f"np ic {ic} \n\n -------pred---------\n {pred} \n\n --------------true-------------\n {true}")
                #     calculate_ic(pred, true)

                total_loss.append(loss.item())
                # total_ic.append(ic[0][1])


        total_loss = np.average(total_loss)
        ic = np.corrcoef(preds,labels)
        MSE = criterion(preds,labels)
        # total_ic = np.average(total_ic)
        # print(f" {ic[0][1]} \n ---------------- IC ----------------- \n {ic} \n ------------MSE-------------\n{MSE}")
        print(f" {ic[0][1]}\n")
        print(f"-----------prediction------- \n{preds}\n")
        print(f"mean of prediction {preds.mean()}, std of prediction {preds.std()}")

        print(f"-----------Lables------- \n{labels}\n")
        print(f"mean of labels {labels.mean()}, std of labels {labels.std()}")

        
        self.model.train()
        return total_loss, ic[0][1]
    
    @staticmethod
    def selective_merge_graphs(global_edge_index, global_edge_attr, dynamic_edge_index, dynamic_edge_attr, num_global_edges=10):
        """
        Selectively merge global and dynamic graphs.
        Args:
            global_edge_index (torch.Tensor): Edge indices for the global graph.
            global_edge_attr (torch.Tensor): Edge attributes for the global graph.
            dynamic_edge_index (torch.Tensor): Edge indices for the dynamic graph.
            dynamic_edge_attr (torch.Tensor): Edge attributes for the dynamic graph.
            num_global_edges (int): Number of global edges to retain.
        Returns:
            torch.Tensor: Merged edge indices and edge attributes.
        """
        # Select top N global edges based on edge attributes
        top_global_edges = torch.topk(global_edge_attr, num_global_edges).indices
        selected_global_edge_index = global_edge_index[:, top_global_edges]
        selected_global_edge_attr = global_edge_attr[top_global_edges]

        # Combine selected global edges with dynamic edges
        edge_index = torch.cat([selected_global_edge_index, dynamic_edge_index], dim=1)
        edge_attr = torch.cat([selected_global_edge_attr, dynamic_edge_attr], dim=0)
        return edge_index, edge_attr

    
    def get_edges_when_training(self,batch,training = True):
        batch = batch.detach().cpu().numpy()
        if self.args.GNN_type ==0:
            return self.edge_index, self.edge_attr
        elif self.args.GNN_type == 1:
            edge_index, edge_attr = mDataset_btc_CGNN.create_graph(batch)
            return edge_index, edge_attr
        elif self.args.GNN_type == 2:
            edge_index, edge_attr = mDataset_btc_CGNN.create_graph(batch)
            if training:
                self.edge_index, self.edge_attr = Exp_Long_Term_Forecast_CGNN.selective_merge_graphs(self.edge_index, self.edge_attr, edge_index, edge_attr)
            return self.edge_index, self.edge_attr
        
    def train(self, setting):

        if self.args.write_graph:
            sample_x= tensorboard_helper.get_sample_input(self.args,self.device)
            self.loger.writer.add_graph(self.model, (sample_x, [],[]))

        if self.args.log_gradient:
            self.loger.log_gradients(self.model)



        print(f"------verbose------- \n {self.args.verbose}")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.model.GNN_type == 0 or self.model.GNN_type == 2:
            self.edge_index, self.edge_attr = train_data.edge_index, train_data.edge_attr
        #Todo: Might be problematic if test without train

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
            for i, (batch_x,batch_y) in enumerate(train_loader):

                # print(f"get from loader {batch_x.shape}")
                self.loger.global_step += 1
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                edge_index, edge_attr = self.get_edges_when_training(batch_x)
                edge_index, edge_attr = edge_index.to(self.device), edge_attr.to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs =  self.model(batch_x, edge_index, edge_attr)[0] # not entered does not matter
                        else:
                            outputs =  self.model(batch_x, edge_index, edge_attr)

                        # f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.label_len:, 0:1]
                        batch_y = batch_y[:, -self.args.label_len:, 0:1].to(self.device)  # only for btc dataset since target is at first column
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # print("in else")
                    if self.args.output_attention:
                        outputs =  self.model(batch_x, edge_index, edge_attr)[0]
                    else:
                        outputs =  self.model(batch_x,edge_index, edge_attr)
                    
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

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

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
            for i, (batch_x,batch_y) in enumerate(test_loader):
                
                edge_index, edge_attr = self.get_edges_when_training(batch_x,training=False)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                edge_index, edge_attr = edge_index.to(self.device), edge_attr.to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs =  self.model(batch_x,edge_index, edge_attr)[0]
                        else:
                            outputs =  self.model(batch_x, edge_index, edge_attr)
                else:
                    if self.args.output_attention:
                        outputs =  self.model(batch_x, edge_index, edge_attr)[0]

                    else:
                        outputs =  self.model(batch_x, edge_index, edge_attr)

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

