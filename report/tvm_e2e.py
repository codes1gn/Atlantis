from tvm import relay, relax
from tvm.relax.testing import relay_translator, nn
import tvm
from mortise.planning.utils import is_function_node
# from collage.cost_func import *
from mortise.planning.optimizer.custom_fusion_pass import CustomFusionPass
from atlantis.models.torch_workloads import get_network_from_torch
from atlantis.models.relay_workloads import get_network_from_relay
from tvm.contrib import graph_executor as runtime
import numpy as np
import argparse
from tvm import autotvm, auto_scheduler
from mortise.planning.utils import *
from atlantis.models.torch_workloads import *
from e2e_perf_logger import *

import time
import os
from utils import *
from mortise.planning.measurer.base import *

from mortise.planning.pattern_manager.pattern_registry import PatternRegistry
# from collage.optimizer._optimizer import visualize_network
from mortise.planning.optimizer.custom_fusion_pass import get_opt_info_tag

OPT_LEVEL = 0
ANSOR_LOG = ""


def setup_attrs_ours(net, net_name, hw_name, batch_size):
    net = net.with_attr("NETWORK_FUNC_ATTR", net_name)
    net = net.with_attr("HW_FUNC_ATTR", hw_name)
    net = net.with_attr("BATCH_SIZE_ATTR", batch_size)

    return net

# No AlterOpLayout
# `AlterOpLayout` pass (enables when `opt_level = 3`) replaces `NCHW` convolution with `NCHW[x]c` implementation on x86 CPUs.


def build_and_measure_autotvm_without_alter_layout(net, params, target_str, shape_dict, hw_name):
    # else:
    with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass=["AlterOpLayout"]):
            lib = relay.build(net, target_str, params=params)
        logging.info(f"We successfully built the network")
        # Create workload
        dev = tvm.device(target_str, 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        for input_name, input_shape in shape_dict.items():
            input_data = np.random.uniform(-1, 1,
                                           size=input_shape).astype("float32")
            module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator(
        "run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)
    mean_perf, std_perf = measure(ftimer, target_str, hw_name)

    return mean_perf, std_perf, module


def build_and_measure_autotvm(net, params, target_str, shape_dict, hw_name):
    # else:
    with autotvm.apply_history_best(get_autotvm_log_path(hw_name)):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL):
            lib = relay.build(net, target_str, params=params)
        logging.info(f"We successfully built the network")
        # Create workload
        dev = tvm.device(target_str, 0)
        module = runtime.GraphModule(lib["default"](dev))

        # Setup execution
        for input_name, input_shape in shape_dict.items():
            input_data = np.random.uniform(-1, 1,
                                           size=input_shape).astype("float32")
            module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator(
        "run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)
    mean_perf, std_perf = measure(ftimer, target_str, hw_name)

    return mean_perf, std_perf, module


def measure_end_to_end_tvm_no_tuning(net, params, target_str, shape_dict, method_mode, net_name, hw_name, batch_size):

    with tvm.transform.PassContext(
        opt_level=0,
        # required_pass=["FastMath"]
    ):
        lib = relay.build(net, target_str, params=params)

    logging.info(f"We successfully built the network")
    # Create workload
    dev = tvm.device(target_str, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1,
                                       size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator(
        "run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)
    mean_perf, std_perf = measure(ftimer, target_str, hw_name)

    return mean_perf, std_perf, module


def measure_end_to_end_tvm_afusion(net, params, target_str, shape_dict, method_mode, net_name, hw_name, batch_size):
    with tvm.transform.PassContext(opt_level=OPT_LEVEL):
        relax_mod = relay_translator.from_relay(
            net, target=target_str, relay_params=params)
        dev = tvm.device(target_str, 0)
        if (target_str == "cuda"):
            seq = tvm.transform.Sequential(
                [relax.transform.LegalizeOps(), tvm.tir.transform.DefaultGPUSchedule()]
            )
            relax_mod = seq(relax_mod)
        ex = relax.build(relax_mod, target_str)
        vm = relax.VirtualMachine(ex, dev)

    logging.info(f"We successfully built the network")
    # Create workload
    relax_mod, params = relax.frontend.detach_params(relax_mod)
    inputs = {}
    for input_name, input_shape in shape_dict.items():
        inputs[input_name] = tvm.nd.array(
            np.random.rand(*input_shape).astype(np.float32), dev)
    input_list = [
        inputs[key.name_hint] for key in relax_mod["main"].params if key.name_hint in inputs
    ]
    if params:
        input_list += params["main"]
    # Setup execution
    # vm.set_input("main", *input_list)
    ftimer = vm.time_evaluator("main", dev)
    # mean_perf, std_perf = measure(ftimer, target_str, hw_name)
    # measure = ftimer(*input_list)
    # mean_perf, std_perf = measure.mean, measure.std
    mean_perf, std_perf = measure(ftimer, target_str, *input_list)
    return mean_perf, std_perf, relax_mod


def measure_end_to_end_perf_autotvm(net, params, target_str, shape_dict, method_mode, net_name, hw_name, batch_size):
    assert is_function_node(net)

    if method_mode is not None:
        net = net.with_attr("CustomFusionPass", method_mode)
        net = setup_attrs_ours(net, net_name, hw_name, batch_size)

    return build_and_measure_autotvm(net, params, target_str, shape_dict, hw_name)


def measure_end_to_end_perf_autosch(net, params, target_str, shape_dict, is_ours, hw_name):
    assert is_function_node(net)

    if is_ours:
        net = net.with_attr("CustomFusionPass", CustomFusionPass.DP)

    with auto_scheduler.ApplyHistoryBest(ANSOR_LOG):
        with tvm.transform.PassContext(opt_level=OPT_LEVEL):
            lib = relay.build(net, target_str, params=params)

    # Create workload
    dev = tvm.device(target_str, 0)
    module = runtime.GraphModule(lib["default"](dev))

    # Setup execution
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1,
                                       size=input_shape).astype("float32")
        module.set_input(input_name, input_data)

    ftimer = module.module.time_evaluator(
        "run", dev, number=NUM_MEASUREMENTS_PER_REPEAT_E2E, repeat=NUM_REPEATS_E2E)
    mean_perf, std_perf = measure(ftimer, target_str, hw_name)

    return mean_perf, std_perf, module


def measure_afusion(mod, params, shape_dict, args, is_perf_logging):
    # For debugging and visualization
    # mod["main"] = mod["main"].with_attr(NETWORK_FUNC_ATTR, args.network)

    mean_perf, std_perf, relax_mod = measure_end_to_end_tvm_afusion(mod["main"], params, args.target, shape_dict,
                                                                    None,
                                                                    args.network, args.hw, args.batch_size)
    print(f"[{args.network}] Performance of afusion on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'AFusion', mean_perf, std_perf, is_perf_logging)


def measure_tvm(mod, params, shape_dict, args, is_perf_logging):
    # For debugging and visualization
    # mod["main"] = mod["main"].with_attr(NETWORK_FUNC_ATTR, args.network)

    mean_perf, std_perf, mod_tvm = measure_end_to_end_tvm_no_tuning(mod, params, args.target, shape_dict,
                                                                    None,
                                                                    args.network, args.hw, args.batch_size)
    print(f"[{args.network}] Performance of TVM on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'TVM', mean_perf, std_perf, is_perf_logging)


def measure_autotvm(mod, params, shape_dict, args, is_perf_logging):
    # For debugging and visualization
    # mod["main"] = mod["main"].with_attr(NETWORK_FUNC_ATTR, args.network)

    mean_perf, std_perf, mod_tvm = measure_end_to_end_perf_autotvm(mod["main"], params, args.target, shape_dict,
                                                                   None,
                                                                   args.network, args.hw, args.batch_size)
    print(f"[{args.network}] Performance of AutoTVM on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'AutoTVM', mean_perf, std_perf, is_perf_logging)


def measure_ansor(mod, params, shape_dict, args, is_perf_logging):
    # For debugging and visualization
    # mod["main"] = mod["main"].with_attr(NETWORK_FUNC_ATTR, args.network)
    print(ANSOR_LOG)
    if (args.tuning == "true"):
        print("Extract tasks...")
        import time
        time1 = time.perf_counter()
        print(time.perf_counter())
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], params, args.target)

        def run_tuning():
            print("Begin tuning...")
            GPU_RUNNER = auto_scheduler.LocalRunner(
                repeat=10, min_repeat_ms=300, timeout=50)
            CPU_RUNNER = auto_scheduler.LocalRunner(
                repeat=10, enable_cpu_cache_flush=True)
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=200,  # change this to 20000 to achieve the best performance
                runner=CPU_RUNNER if args.hw == "cpu" else GPU_RUNNER,
                measure_callbacks=[auto_scheduler.RecordToFile(ANSOR_LOG)],
            )
            tuner.tune(tune_option)
        print(time.perf_counter())

        run_tuning()
        time2 = time.perf_counter()
        print(time.perf_counter())
        args.tune_time = time2 - time1
    mean_perf, std_perf, mod_tvm = measure_end_to_end_perf_autosch(mod["main"], params, args.target, shape_dict,
                                                                   False, args.hw)
    print(f"[{args.network}] Performance of Ansor on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_e2e_perf(args, 'Ansor', mean_perf, std_perf, is_perf_logging)


def verify_network_output(net, shape_dict, mod_tvm, mod_ours):
    assert is_function_node(net)

    # Create same input data for two networks
    name_to_data = {}
    for input_name, input_shape in shape_dict.items():
        input_data = np.random.uniform(-1, 1,
                                       size=input_shape).astype("float32")
        name_to_data[input_name] = input_data

    # Setup execution
    for input_name, input_data in name_to_data.items():
        mod_tvm.set_input(input_name, input_data)

    mod_tvm.run()
    out_tvm = mod_tvm.get_output(0).asnumpy()

    # Setup execution
    for input_name, input_data in name_to_data.items():
        mod_ours.set_input(input_name, input_data)

    mod_ours.run()
    out_ours = mod_ours.get_output(0).asnumpy()

    TOL = 1e-01
    print("First 10 outputs")
    print(f"TVM    : {out_tvm.flatten()[:10]}")
    # print(f"AutoTVM: {out_tvm.flatten()[:10]}")
    print(f"Ours   : {out_ours.flatten()[:10]}")
    assert np.allclose(out_tvm, out_ours, rtol=TOL, atol=TOL)

    print(f"Passed the verification of output test")
    print(f"Worst diffence : {np.abs((out_ours - out_tvm)).max():.4f}")


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
    parser.add_argument("-r", "--runner", default="naive", help="runner")
    parser.add_argument("-t", "--tuning", default="false", help="tuning?")
    # parser.add_argument("-t", "--target", help="target device")
    # parser.add_argument("-dt", "--dtype", help="data type")

    args = parser.parse_args()

    args_checker(args, parser)
    return args


if __name__ == "__main__":
    args = get_args()
    # Redirect output to log files
    log_dir = "e2e_measure_logs"
    if args.network == "nasneta":
        OPT_LEVEL == 2
    args.target = get_build_target(args.hw)
    print(args.target)
    # For tuning time measurement, comment setup_logging above and uncomment the following codes
    # logging.basicConfig(level=logging.ERROR)

    # It shows all logs. Still, it is too messy though cuz TVM logs are interrupting with our logs
    logging.basicConfig(level=logging.INFO)
    # logging.basicConfig(level=logging.WARNING)

    # We can't test this because this network include batch norm.
    logging.info(f"batch size: {args.batch_size}")

    mod, params, shape_dict, _ = get_network_from_torch(
        args.network, args.batch_size)

    # Debugging Yolo-v3
    # from tvm.relay.transform.utility.visualize import visualize_network
    # visualize_network(mod["main"], "o3_yolov3")

    # mod, params, shape_dict, _ = get_network_from_torch("nasneta", 1)
    # mod, params, shape_dict, _ = get_network_from_relay("conv2d", 1)
    # mod, params, shape_dict, _ = get_network_from_relay("conv2d+relu_x2", 1)
    # mod, params, shape_dict, _ = get_network_from_relay("diamond", 1)
    # mod, params, shape_dict, _ = get_network_from_relay("reshape", 1)
    # Debugging for BERT_full (only including first block)
    # mod, params, shape_dict, _ = crop_network_from_torch(args.network, 1, 100)

    # Assign build target based on a given hw
    args.target = get_build_target(args.hw)
    is_perf_logging = True
    # is_perf_logging = False
    if (args.runner == "naive"):
        logging.info(f"进入naive: {args}")
        measure_tvm(mod, params, shape_dict, args, is_perf_logging)
    if (args.runner == "afusion"):
        logging.info(f"进入afusion: {args}")
        measure_afusion(mod, params, shape_dict, args, is_perf_logging)
    elif args.runner == "autotvm":
        logging.info(f"进入autotvm: {args}")
        measure_autotvm(mod, params, shape_dict, args, is_perf_logging)
    elif args.runner == "ansor":
        logging.info(f"进入Ansor: {args}")
        ANSOR_LOG = BASE_PATH + \
            "ansor-%s-B%d-%s.json" % (args.network, args.batch_size, args.hw)
        measure_ansor(mod, params, shape_dict, args, is_perf_logging)

    # mean_perf, std_perf, mod_dp = measure_end_to_end_perf_autotvm(mod["main"], params, args.target, shape_dict,
    #                                                              CustomFusionPass.DP,
    #                                                              args.network, args.hw, args.batch_size)
    # print(f"[{args.network}] Performance of DP on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    # print("NETWORK LOADED")
    # mean_perf, std_perf, mod_dnnl = measure_end_to_end_perf_dnnl(mod, params, args.target, shape_dict, args.hw, args)
    # print(f"[{args.network}] Performance of DNNL on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    # mean_perf, std_perf, mod_dp = measure_end_to_end_perf_autotvm(mod["main"], params, args.target, shape_dict,
    #                                                               CustomFusionPass.DP,
    #                                                               args.network, args.hw, args.batch_size)
    # print(f"[{args.network}] Performance of DP on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    # mean_perf, std_perf, mod_cud = measure_end_to_end_perf_single_backend_without_alter_layout(mod["main"], params, args.target, shape_dict,
    #                                                                          args.network, args.hw, args.batch_size,
    #                                                                          Target.MKL.id())
    # print(f"[{args.network}] Performance of MKL on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    # print(f"[{args.network}] Performance of TVM (no tuning) on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")

    # verify_network_output(mod["main"], shape_dict, mod_tvm, None)

    # measure_dp_and_baselines(mod, params, shape_dict, args, is_perf_logging)

    # measure_two_level(mod, params, shape_dict, args, is_perf_logging)
    # measure_dp_tuning_time(mod, params, shape_dict, args, is_perf_logging)

    # Debug: test single backend pipeline that offloads ops to single backend whenever possible
    # single_backend = Target.CUDNN
    # measure_single_backend_debug(mod, params, shape_dict, args, is_perf_logging, single_backend)

    # Note that this one do not use AutoTVM because cudnn and cublas will be used only if AutoTVM is disabled
    # if args.hw in NVIDIA_GPUS:
    #    measure_tvm_strategy_libs(mod, params, 'cuda -libs=cudnn,cublas', shape_dict, args, is_perf_logging)
    # elif args.hw in INTEL_CPUS:
    #    measure_tvm_strategy_libs(mod, params, 'llvm -libs=mkl', shape_dict, args, is_perf_logging)
    # else:
    #    raise Exception(f"{args.hw} is unexpected hw, we need to set default backends for this hw.")

    # NasNet-A only works for opt_level 2 (not 3 due to the avgpool2d issue)
    # if args.network == "nasneta":
    #     OPT_LEVEL.set(2)
