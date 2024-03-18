"""
Post Training Quantization: After the model is trained. It goes from floating point to interger 8 fore example
Flow: (Pretrained model + Calibration data) -> Calibration -> Quantization -> Quantized model 
Advatange: No need to retrain the model
Disadvantage: Performance can be poor


Quatization Aware Training: Training the model with quantization in mind
Flow: ((Pretrained model -> Quantization) + Training data ) -> Retraining/Finetuning -> Quantized Model
Advantage: Better performance
Disadvantage: Complex, multiple retraining runs

Uniform Quantization: Applies equal spacing between quantization levels
Symmetric Uniform Quantization: Applies equal spacing between quantization levels, but the zero point is in the middle of the range
Asymmetric Uniform Quantization: Applies equal spacing between quantization levels, but the zero point is not in the middle of the range
Non-Uniform Quantization: Applies unequal spacing between quantization levels

Layer-Wise Quantization: Applies quantization to each layer of the model
Channel-Wise Quantization: Applies quantization to each channel of each layer of the model

Ternary Quantization: Quantization to 3 levels

Mixed-Precision: Using different precisions for different parts of the model

Knowledge Distillation: Training a smaller model to mimic the larger model

In Pytorch:
- Post training dynamic/weight_only qunatization
- Post training static
- Quantization aware training
"""

import torch
import os
from torch.fx import symbolic_trace


from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx

# from torch.ao.quantization import FakeQuantize

# from torch.ao.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
# from torch.ao.quantization.observer import PerChannelMinMaxObserver, MovingAveragePerChannelMinMaxObserver

from utils.train_validation_testing import calibrate_model_ptq, train_model

def print_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    model_size_string = f"Size (MB): {os.path.getsize('temp.p')/1e6}"
    os.remove('temp.p')
    return model_size_string

def get_fakequantize(bits: int, scheme: str, observer: str, ch_axis: [int, None]=None):
    """

    :param bits: number of bits
    :param scheme: "per_tensor" or "per_channel"
    :param observer: MinMax or Moving Average MinMax
    :param ch_axis: 0 for weights (default).
    :return:
    """
    kwargs = {
        'quant_min': -2 ** (bits - 1),
        'quant_max': 2 ** (bits - 1) - 1,
        'dtype': torch.qint8,
        'reduce_range': False,
    }

    if scheme == 'per_tensor':
        qscheme = torch.per_tensor_symmetric
        observer_dict = {
            'minmax': torch.ao.quantization.MinMaxObserver,
            'maminmax': torch.ao.quantization.MovingAverageMinMaxObserver,
        }
        observer = observer_dict[observer]

    elif scheme == 'per_channel':
        qscheme = torch.per_channel_symmetric
        observer_dict = {
            'minmax': torch.ao.quantization.PerChannelMinMaxObserver,
            'maminmax': torch.ao.quantization.MovingAveragePerChannelMinMaxObserver,
        }
        observer = observer_dict[observer]
        kwargs['ch_axis'] = 0 if qscheme == torch.per_channel_symmetric and ch_axis is None else ch_axis
    else:
        raise NotImplementedError

    kwargs['qscheme'] = qscheme
    kwargs['observer'] = observer

    fake_quantization = torch.ao.quantization.FakeQuantize.with_args(**kwargs)

    return torch.ao.quantization.QConfig(activation=fake_quantization, weight=fake_quantization)

# def create_custom_qconfig(bits: int, scheme: str, observer: str, ch_axis: [int, None] = None):
#     # Use the get_fakequantize function to create the fake quantize module
#     fake_quantize_module = get_fakequantize(bits, scheme, observer, ch_axis)
#     # Assuming you are quantizing both weights and activations, you might need separate configurations
#     # Here, for simplicity, we use the same for both
#     return torch.ao.quantization.QConfig(activation=fake_quantize_module, weight=fake_quantize_module)


class QuantizationMarshalling():

    """
    Default quantization packages: x86(old->fbgemm), qnnpack
    """

    def __init__(self, args, file_stats, train_logs, date_time):
        self.args = args
        self.file_stats = file_stats
        self.train_logs = train_logs
        self.date_time = date_time

    def ptq(self, model ,configuration, dataloader_train, args):

        print(f"Model size before quantization: {print_model_size(model)}")

        model.to('cpu')
        model.eval()

        if configuration['native_backend'] == 'custom':
            # qconfig_mapping = create_custom_qconfig(configuration['custom_qconfig']['bits'], configuration['custom_qconfig']['scheme'], configuration['custom_qconfig']['observer'], configuration['custom_qconfig']['ch_axis'])
            qconfig_mapping = get_fakequantize(
                bits=configuration['custom_qconfig']['bits'],
                scheme=configuration['custom_qconfig']['scheme'],
                observer=configuration['custom_qconfig']['observer'],
                ch_axis=configuration['custom_qconfig']['ch_axis']
            )
            qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig_mapping)
        else:
            qconfig_mapping = get_default_qconfig_mapping('x86')

        example_inputs = dataloader_train
        model_prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs)

        model_calibrated = calibrate_model_ptq(model_prepared, dataloader_train, args)

        model_quantized = quantize_fx.convert(model_calibrated)   

        print("Quantized model ready")
        print(f"Model size after quantization: {print_model_size(model_quantized)}")

        return model_quantized

    def qat(self, model ,configuration, dataloader_train, optimizer, args):

        print("Starting quantize aware training")
        print(f"Model size before quantization: {print_model_size(model)}")

        model.to('cpu')
        model.train()

        symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
        # print(symbolic_traced.graph)
        # print(symbolic_traced.code)

        if configuration['native_backend'] == 'custom':
            # qconfig_mapping = create_custom_qconfig(configuration['custom_qconfig']['bits'], configuration['custom_qconfig']['scheme'], configuration['custom_qconfig']['observer'], configuration['custom_qconfig']['ch_axis'])
            qconfig_mapping = get_fakequantize(
                bits=configuration['custom_qconfig']['bits'],
                scheme=configuration['custom_qconfig']['scheme'],
                observer=configuration['custom_qconfig']['observer'],
                ch_axis=configuration['custom_qconfig']['ch_axis']
            )
            qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig_mapping)
        else:
            qconfig_mapping = get_default_qconfig_mapping('x86')
            # qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global('x86')
        
        # prepare for QAT
        example_inputs = dataloader_train

        model_prepared = quantize_fx.prepare_qat_fx(model, qconfig_mapping, example_inputs)

        epochs = configuration['number_of_epochs_qat']

        for epoch in range(epochs):
            train_model(model_prepared, dataloader_train, optimizer, args, self.file_stats, self.train_logs, epoch, qat = True)

        model_quantized = quantize_fx.convert(model_prepared)

        print("Quantized model ready")
        print(f"Model size after quantization: {print_model_size(model_quantized)}")

        return model_quantized

    def quantization_prepare_ptq():
        pass
