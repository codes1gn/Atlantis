import torch.nn as nn
import torch
import math
import argparse
from tqdm import tqdm
# from tvm.relay.transform.pattern_manager.target import *

from atlantis.models.torch_workloads import *
from atlantis.models import *
# import tensorflow as tf
# import time
# from measure_end_to_end import log_e2e_perf
import time
from e2e_perf_logger import *

from torchsummary import summary


def log_e2e_perf(args, method, mean_perf, std_perf, is_perf_logging):
    if is_perf_logging:
        E2EPerfLogger().log_perf(args.hw, args.batch_size,
                                 args.network, method, mean_perf, std_perf)


def args_checker(args, parser):
    is_missing_arg = not args.network
    is_missing_arg |= not args.hw
    # is_missing_arg |= not args.batch_size
    # is_missing_arg |= not args.target
    # is_missing_arg |= not args.dtype

    if is_missing_arg:
        parser.error('Make sure you input all arguments')


def get_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1,
                        type=int, help="batch size")
    parser.add_argument('-tensorrt', action='store_true')

    # Measurement related parameters
    # parser.add_argument("--iterations", help="How many iterations to average for timing", type=int, default=10000)
    # parser.add_argument("--discard_iter", help="How many iterations to not time during warm up", type=int, default=2000)
    parser.add_argument(
        "--iterations", help="How many iterations to average for timing", type=int, default=100)
    parser.add_argument(
        "--discard_iter", help="How many iterations to not time during warm up", type=int, default=20)

    args = parser.parse_args()

    args_checker(args, parser)
    return args


def get_torch_model_and_input(args):
    model = NETWORK_TO_TORCH_MODEL[args.network]()
    inputs = get_torch_input_data(args.network, args.batch_size)

    if args.hw == 'gpu':
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    return model, inputs


def measure_torch(model, inputs, args, is_perf_logging):
    times = []
    # t = 0
    with torch.no_grad():
        for i in tqdm(range(args.discard_iter + args.iterations)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            # t1 = time.perf_counter()
            model(inputs)
            # t2 = time.perf_counter()
            end.record()
            # Waits for everything to finish running
            torch.cuda.synchronize()
            # t2 = time.perf_counter()
            # t = t+ t2-t1
            times.append(start.elapsed_time(end))
    times = np.array(times)[args.discard_iter:]
    print(args.discard_iter + args.iterations)
    mean_perf, std_perf = np.mean(times), np.std(times)
    print(f"[{args.network}] Performance of PyTorch1 on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'PyTorch1', mean_perf, std_perf, is_perf_logging)

# Link: https://discuss.pytorch.org/t/how-to-measure-execution-time-in-pytorch/111458
# Note: CPU operations are synchronous; you can use any Python runtime profiling method like time.time().


def measure_torch_cpu(model, inputs, args, is_perf_logging):
    times = []
    with torch.no_grad():
        for i in tqdm(range(args.discard_iter + args.iterations)):
            start_time = time.time()
            model(inputs)
            times.append((time.time() - start_time)*1000.0)

    times = np.array(times)[args.discard_iter:]
    mean_perf, std_perf = np.mean(times), np.std(times)
    print(f"[{args.network}] Performance of PyTorch1 on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'PyTorch1', mean_perf, std_perf, is_perf_logging)


def measure_trt(model, inputs, args, is_perf_logging):
    from torch2trt import torch2trt
    import time

    # summary(model, (64, 256))

    model_trt = torch2trt(model, [inputs])

    times = []
    for i in tqdm(range(args.discard_iter + args.iterations)):
        torch.cuda.current_stream().synchronize()
        t0 = time.time()
        model_trt(inputs)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        times.append(1000.0 * (t1 - t0))

    times = np.array(times)[args.discard_iter:]
    mean_perf, std_perf = np.mean(times), np.std(times)
    print(f"[{args.network}] Performance of TensorRT on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'TensorRT', mean_perf, std_perf, is_perf_logging)


if __name__ == '__main__':
    args = get_args()

    is_perf_logging = True

    model, inputs = get_torch_model_and_input(args)
    if not args.tensorrt:
        # PyTorch measurement
        if args.hw == "gpu":
            measure_torch(model, inputs, args, is_perf_logging)
        elif args.hw == "cpu":
            measure_torch_cpu(model, inputs, args, is_perf_logging)
        elif args.hw == "trt":
            measure_trt(model, inputs, args, is_perf_logging)
        else:
            raise Exception(
                f"{args.hw} is unexpected hw, we need to set default backends for this hw.")
    else:
        measure_trt(model, inputs, args, is_perf_logging)
