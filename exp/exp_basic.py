import os
import torch
from models import Autoformer, PatchMixer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, TemporalFusionTransformer, miTransformer
from models.FFNs import DTFFN, DTM_FFN, FFN, AC_FFN, DFFN, FAC_DFFN, GNN_LSTM_FFN, GTFFN, LSTM_FFN, MFFN, MFFN_D, SAC_DFFN, SAC_DFFN_D, CGDT_FFN_dynamic, CGDT_FFN_hybird, CGDT_FFN_static, DTFFN_Ablation, DTFFN_abspooling, DTFFN_pooling, DTFFN_pooling2d, DTM_FFN_crossatten, DTM_FFN_dual, DTM_FFN_m, FFN_2x1, FFN_2x1_avg, FFN_2x1_max, FFN_2x1_max_D, FFN_2x2, FFN_sac, LSTM_FFN_ablation, LSTM_FFN_sur

from models.MLPs import  Baseline, Baseline_20, Baseline_10, Baseline_15, Baseline_20_m, Baseline_25, Baseline_25_m, Baseline_25_osci, Baseline_5
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'miTransformer': miTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            # 'MambaSimple': MambaSimple,
            # 'Mamba': Mamba,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "PatchMixer": PatchMixer,

            "FFN_2x1": FFN_2x1,
            "FFN_2x1_avg": FFN_2x1_avg,
            "FFN_2x1_max": FFN_2x1_max,
            "FFN_2x1_max_D": FFN_2x1_max_D,


            "FFN_sac": FFN_sac,
            "FFN_2x2": FFN_2x2,


            "FFN": FFN,
            "AC_FFN": AC_FFN,
            "DFFN": DFFN,
            
            "MFFN": MFFN,
            "MFFN_D": MFFN_D,

            "GTFFN": GTFFN,


            "DTFFN": DTFFN,
            "DTFFN_Ablation": DTFFN_Ablation,
            "DTM_FFN": DTM_FFN,
            "DTM_FFN_m": DTM_FFN_m,
            "DTM_FFN_dual": DTM_FFN_dual,
            "DTM_FFN_crossatten": DTM_FFN_crossatten,
            "DTFFN_pooling": DTFFN_pooling,
            "DTFFN_pooling2d": DTFFN_pooling2d,

            "DTFFN_abspooling": DTFFN_abspooling,


            "CGDT_FFN_static": CGDT_FFN_static, 
            "CGDT_FFN_dynamic": CGDT_FFN_dynamic, 
            "CGDT_FFN_hybird": CGDT_FFN_hybird, 

            "GNN_LSTM_FFN": GNN_LSTM_FFN, 

            "SAC_DFFN": SAC_DFFN,
            "SAC_DFFN_D": SAC_DFFN_D,
            "FAC_DFFN": FAC_DFFN,

            "LSTM_FFN":LSTM_FFN,
            "LSTM_FFN_ablation":LSTM_FFN_ablation,
            "LSTM_FFN_sur":LSTM_FFN_sur,

            "Baseline": Baseline,
            "Baseline_5": Baseline_5,
            "Baseline_10": Baseline_10,
            "Baseline_15": Baseline_15,
            "Baseline_20": Baseline_20,
            "Baseline_20_m": Baseline_20_m,
            "Baseline_25": Baseline_25,
            "Baseline_25_m": Baseline_25_m,
            "Baseline_25_osci": Baseline_25_osci

        }
        if not self.args.tf:
            self.device = self._acquire_device()
            self.model = self._build_model().to(self.device)
        else:
            self.device = self._acquire_device()
            self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
