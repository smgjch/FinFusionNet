import torch
import torch.nn as nn

def get_sample_input(args,device):
    # Adjust these dimensions based on your actual input shape
    batch_size = args.batch_size
    seq_len =  args.seq_len
    feature_dim =  args.enc_in
    
    sample_x = torch.randn(batch_size, seq_len, feature_dim).to( device)
    sample_x_mark = torch.randn(batch_size, seq_len, 1).to( device)
    sample_y = torch.randn(batch_size, seq_len, 1).to( device)
    sample_y_mark = torch.randn(batch_size, seq_len,1).to( device)


    # print(f"shape {sample_x.shape}")
    return sample_x, sample_x_mark, sample_y, sample_y_mark

class Loger:
    def __init__(self,global_step, writer) -> None:
        self.global_step = global_step
        self.writer = writer
        self.freq = 100


    def gradient_log_hook(self, name):
        def hook(module, grad_input, grad_output):
            if self.global_step % self.freq == 0:
                self.writer.add_histogram(f"{name}_grad", grad_output[0], self.global_step)
        return hook

    def log_gradients(self, model):
        for name, layer in model.named_modules():

            layer.register_backward_hook(self.gradient_log_hook(name))
