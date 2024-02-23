import torch
import os
import pynvml
from timm.models import create_model
from timm.data import resolve_data_config
from timm.optim import create_optimizer_v2
import time
import numpy as np
from utils import logger, CSV_RW


class Trainer:
    def __init__(self, model_name, batch_size, opt, lr, weight_decay, drop_rate, model_dtype, device, num_warm_iter, num_bench_iter):
        self.model_name = model_name
        self.opt = opt
        self.lr = lr
        self.weight_decay = weight_decay
        self.drop_rate = drop_rate
        self.model_dtype = model_dtype
        self.data_dtype = model_dtype
        self.device = device
        self.batch_size = batch_size
        self.target_shape = tuple()

        self.num_warm_iter = num_warm_iter
        self.num_bench_iter = num_bench_iter

        self.model = torch.nn.Module()
        self.num_classes = []
        self.input_size = []
        self.dummy_inputs = []

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.train_results = dict()
        self.inference_results = dict()
        self.final_results = dict()
        self.onnx_file = ""
        self.file_name = ""

    def _init_model(self, torchscript=False):
        self.model = create_model(
            self.model_name, pretrained=False, in_chans=3, exportable=True, drop_rate=self.drop_rate, scriptable=torchscript)
        self.model.to(device=self.device, dtype=self.model_dtype)
        data_config = resolve_data_config({}, model=self.model)
        self.input_size = data_config['input_size']
        self.num_classes = self.model.num_classes

    def _init_input(self):
        self.dummy_inputs = torch.randn(
            (self.batch_size,) + self.input_size, device=self.device, dtype=self.data_dtype)  # type: ignore

    def _gen_target(self, batch_size):
        return torch.empty(
            (batch_size,) + self.target_shape, device=self.device, dtype=torch.long).random_(self.num_classes)  # type: ignore

    def _get_gpu_memory(self):
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        memory_usage = memory_info.used / 1024 / 1024  # type: ignore
        return memory_usage

    def _get_gpu_util(self):
        utilisaton = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return (utilisaton.gpu, utilisaton.memory)

    def _get_gpu_power(self):
        try:
            power_info = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_usage = power_info / 1000  # convert to watts
            return power_usage
        except:
            # logger.info("Power usage not supported")
            return 0

    def train(self):
        torch.cuda.empty_cache()
        self._init_model()
        self.model.train()
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = create_optimizer_v2(
            self.model,
            opt=str(self.opt),
            lr=int(self.lr),
            weight_decay=float(self.weight_decay))
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def _step():
            self.optimizer.zero_grad()

            start_event.record()  # type: ignore
            output = self.model(self.dummy_inputs)  # type: ignore
            if isinstance(output, tuple):
                output = output[0]
            end_event.record()  # type: ignore
            torch.cuda.synchronize()
            delta_fwd = start_event.elapsed_time(end_event)

            target = self._gen_target(output.shape[0])
            self.loss(output, target).backward()
            end_event.record()  # type: ignore
            torch.cuda.synchronize()
            delta_bwd = start_event.elapsed_time(end_event) - delta_fwd

            self.optimizer.step()
            end_event.record()  # type: ignore
            torch.cuda.synchronize()
            delta_opt = start_event.elapsed_time(
                end_event) - delta_bwd - delta_fwd

            gpu_memory = self._get_gpu_memory()
            gpu_power = self._get_gpu_power()
            gpu_util, memory_util = self._get_gpu_util()
            return delta_fwd, delta_bwd, delta_opt, gpu_memory, gpu_power, gpu_util, memory_util

        self._init_input()

        for _ in range(self.num_warm_iter):
            _step()

        t_run_start = time.perf_counter()

        total_fwd = []
        total_bwd = []
        total_opt = []
        total_step = []
        gpu_power_w = []
        gpu_memory_mb = []
        gpu_utilsation = []
        mem_utilsation = []
        num_samples = 0
        for _ in range(self.num_bench_iter):
            delta_fwd, delta_bwd, delta_opt, MB, W, gpu_util, mem_util = _step()
            num_samples += self.batch_size
            total_fwd.append(round(delta_fwd, 3))
            total_bwd.append(round(delta_bwd, 3))
            total_opt.append(round(delta_opt, 3))
            total_step.append(round((delta_fwd + delta_bwd + delta_opt), 3))
            gpu_power_w.append(W)
            gpu_memory_mb.append(MB)
            gpu_utilsation.append(gpu_util)
            mem_utilsation.append(mem_util)
        torch.cuda.empty_cache()

        t_run_elapsed = time.perf_counter() - t_run_start
        self.train_results = dict(
            train_samples_per_sec=round(num_samples / t_run_elapsed, 2),
            train_step_time=round(np.mean(total_step), 3),
            train_step_time_std=round(np.std(total_step), 3),
            train_fwd_time=round(np.mean(total_fwd), 3),
            train_fwd_time_std=round(np.std(total_fwd), 3),
            train_bwd_time=round(np.mean(total_bwd), 3),
            train_bwd_time_std=round(np.std(total_bwd), 3),
            train_opt_time=round(np.mean(total_opt), 3),
            train_opt_time_std=round(np.std(total_opt), 3),
            train_gpu_power_w=round(np.mean(gpu_power_w), 3),
            train_gpu_power_w_std=round(np.std(gpu_power_w), 3),
            train_gpu_memory_mb=round(np.mean(gpu_memory_mb), 3),
            train_gpu_memory_mb_std=round(np.std(gpu_memory_mb), 3),
            train_gpu_utilsation=round(np.mean(gpu_utilsation), 3),
            train_gpu_utilsation_std=round(np.std(gpu_utilsation), 3),
            train_mem_utilsation=round(np.mean(mem_utilsation), 3),
            train_mem_utilsation_std=round(np.std(mem_utilsation), 3),


            train_batch_size=self.batch_size,
        )

        return self.train_results

    def inference(self, mode='torchscript'):
        import torch
        torch.cuda.empty_cache()
        self._init_model()
        self.model.eval()
        if mode == 'torchscript':
            self.model = torch.jit.script(self.model)
        elif mode == 'compile':
            import torch._dynamo
            torch._dynamo.reset()
            self.model = torch.compile(self.model, backend="inductor")
        else:
            pass

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        def _step():
            torch.cuda.empty_cache()
            start_event.record()  # type: ignore
            _ = self.model(self.dummy_inputs)
            end_event.record()  # type: ignore
            torch.cuda.synchronize()
            t_run_elapsed = start_event.elapsed_time(end_event)
            gpu_memory = self._get_gpu_memory()
            gpu_power = self._get_gpu_power()
            gpu_util, memory_util = self._get_gpu_util()
            return t_run_elapsed, gpu_memory, gpu_power, gpu_util, memory_util

        with torch.no_grad():
            self._init_input()

            for _ in range(self.num_warm_iter):
                _step()

            total_step = []
            num_samples = 0
            gpu_power_w = []
            gpu_memory_mb = []
            gpu_utilsation = []
            mem_utilsation = []
            t_run_start = time.perf_counter()
            for _ in range(self.num_bench_iter):
                delta_fwd, MB, W, gpu_util, memory_util = _step()
                total_step.append(round(delta_fwd, 3))
                num_samples += self.batch_size
                gpu_power_w.append(W)
                gpu_memory_mb.append(MB)
                gpu_utilsation.append(gpu_util)
                mem_utilsation.append(memory_util)
            t_run_end = time.perf_counter()
            t_run_elapsed = t_run_end - t_run_start
            torch.cuda.empty_cache()

        self.inference_results = dict(
            inference_samples_per_sec=round(num_samples / t_run_elapsed, 2),
            inference_step_time=round(np.mean(total_step), 3),
            inference_step_time_std=round(np.std(total_step), 3),
            inference_batch_size=self.batch_size,
            inference_gpu_power_w=round(np.mean(gpu_power_w), 3),
            inference_gpu_power_w_std=round(np.std(gpu_power_w), 3),
            inference_gpu_memory_mb=round(np.mean(gpu_memory_mb), 3),
            inference_gpu_memory_mb_std=round(np.std(gpu_memory_mb), 3),
            inference_gpu_utilsation=round(np.mean(gpu_utilsation), 3),
            inference_gpu_utilsation_std=round(np.std(gpu_utilsation), 3),
            inference_mem_utilsation=round(np.mean(mem_utilsation), 3),
            inference_mem_utilsation_std=round(np.std(mem_utilsation), 3),
        )

        return self.inference_results
    
    def get_results(self):
        # combine train and inference results
        results = {}
        results['model_name'] = self.model_name
        results['model_dtype'] = str(self.model_dtype).split('.')[-1]
        results['batch_size'] = self.batch_size
        results['opt'] = self.opt
        results['lr'] = self.lr
        results['weight_decay'] = self.weight_decay
        results["drop_rate"] = self.drop_rate
        results['input_size'] = '_'.join(map(str, self.input_size)) # type: ignore
        results = {**results, **self.train_results, **self.inference_results}
        results["onnx_file"] = self.onnx_file # type: ignore

        logger.info(f'Finished benchmarking {self.file_name} with results: {results}')
        return results
    
    def save_results(self, csv_file):
        results = self.get_results()
        csv = CSV_RW(csv_file)
        csv.write_csv(results)

    
    def export_onnx(self, path):
        # name =  f'{self.model_name}_{self.batch_size}_{self.opt}_{self.lr}_{self.weight_decay}_{self.weight_decay}_{self.model_dtype}_{self.input_size[-1]}'
        name =  f'{self.model_name}_{self.batch_size}'
        self.file_name = name.replace('.', '-') + '.onnx'
        self.onnx_file = os.path.join(path, self.file_name)  
        logger.info(self.onnx_file, os.path.exists(self.onnx_file))
        if not os.path.exists(self.onnx_file):
            torch.cuda.empty_cache()
            self._init_model()
            with torch.no_grad():
                self._init_input()
                self.model.eval()
                torch.onnx.export(self.model, self.dummy_inputs, self.onnx_file, verbose=True, export_params=False) # type: ignore
            torch.cuda.empty_cache()

    def run(self, csv_file, onnx_path):
        try:
            import gc
            gc.collect()
            self.train()
            gc.collect()
            self.inference()
            gc.collect()
            self.export_onnx(onnx_path)
            gc.collect()
            self.save_results(csv_file)
            return True
        except Exception as e:
            logger.error(f'Failed to benchmark {self.model_name} with error: {e}')
            return False
        
    def export_all_models(self, onnx_path):
        try:
            import gc
            gc.collect()
            self.export_onnx(onnx_path)
            return True
        except Exception as e:
            logger.error(f'Failed to export {self.model_name} with error: {e}')
            return False
        


