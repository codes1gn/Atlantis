from tvm import relay
import tvm
from planning.utils import is_function_node
# from collage.cost_func import *
from planning.optimizer.custom_fusion_pass import CustomFusionPass
from atlantis.models.torch_workloads import get_network_from_torch
from atlantis.models.relay_workloads import get_network_from_relay
from tvm.contrib import graph_executor as runtime
import numpy as np
import argparse
from tvm import autotvm, auto_scheduler
from planning.utils import *
from atlantis.models.torch_workloads import *
from e2e_perf_logger import *

import time
import os
from utils import *
from planning.measurer.base import *

from planning.pattern_manager.pattern_registry import PatternRegistry
# from collage.optimizer._optimizer import visualize_network
from planning.optimizer.custom_fusion_pass import get_opt_info_tag

from tvm_e2e import *
from tensorrt_e2e import *
from tvm.ir.transform import PassContext
from tvm import dlight as dl
from tvm import meta_schedule as ms
from tvm.meta_schedule import schedule_rule, postproc
import json
OPT_LEVEL = 3
ANSOR_LOG = ""
database_dir = "/tune_tmp"
workspace_dir = "/root/wang/afusion/log/relax/"
cuda = tvm.target.Target("cuda")

sch_rules = [
    schedule_rule.ApplyCustomRule(),
    multi_level_tiling_tensor_core(),
]

postprocs = [
    postproc.RewriteParallelVectorizeUnroll(),
    postproc.RewriteReductionBlock(),
    postproc.RewriteTensorize(vectorize_init_loop=True),
]


def get_args():
    parser = argparse.ArgumentParser()
    # Default type is string for argparse
    parser.add_argument("-n", "--network", help="name of a neural network")
    parser.add_argument("-hw", "--hw", help="target hardware")
    parser.add_argument("-bs", "--batch-size", default=1,
                        type=int, help="batch size")
    parser.add_argument("-r", "--runner", default="naive", help="runner")
    parser.add_argument("-t", "--tuning", default="false", help="tuning?")
    parser.add_argument("-f", "--func", default="model",
                        help="model or subgraph?")
    # parser.add_argument("-t", "--target", help="target device")
    # parser.add_argument("-dt", "--dtype", help="data type")

    args = parser.parse_args()

    args_checker(args, parser)
    return args


def setup_attrs_ours(net, net_name, hw_name, batch_size):
    net = net.with_attr("NETWORK_FUNC_ATTR", net_name)
    net = net.with_attr("HW_FUNC_ATTR", hw_name)
    net = net.with_attr("BATCH_SIZE_ATTR", batch_size)

    return net

# stage 1: graph opt fuse
# stage 2: graph opt and memory opt  dlight
# stage 3: graph opt and library  collage
# stage 1: graph opt and memory opt and library  collage(relay) => dlight(relax)
# stage 1: graph opt and memory opt and library and microkernel  instrins ms


def measure_relax_dlight(stage, net, params, target_str, shape_dict, method_mode, net_name, hw_name, batch_size):
    with tvm.transform.PassContext(opt_level=OPT_LEVEL):
        path = workspace_dir+net_name+".so"
        param_path = f'{workspace_dir}{net_name}.params'
        print(path)
        dev = tvm.device(target_str, 0)
        compile_time = 0
        # if(not os.path.isfile(path)):
        if True:
            print("not exist")
            relax_mod = relay_translator.from_relay(
                net, target=target_str, relay_params=params)
            t1 = time.perf_counter()
            if ("cuda" in target_str):
                with cuda:
                    seq = tvm.transform.Sequential(
                        [relax.transform.LegalizeOps(
                        ), tvm.tir.transform.DefaultGPUSchedule()]
                    )
                    relax_mod = seq(relax_mod)

                    seq = tvm.transform.Sequential(
                        [
                            relax.transform.DecomposeOpsForInference(),
                            relax.transform.LegalizeOps(),
                            relax.transform.AnnotateTIROpPattern(),
                            relax.transform.FuseOps(),
                            relax.transform.FuseTIR()
                        ]
                    )
                    relax_mod = seq(relax_mod)

                    with cuda:
                        relax_mod = dl.ApplyDefaultSchedule(
                            # dl.gpu.Matmul(),
                            # dl.gpu.Transpose(),
                            # dl.gpu.Reduction(),
                            # dl.gpu.Transpose(),
                            dl.gpu.DecodeGEMV(),
                            # dl.gpu.Matmul(),
                            # dl.gpu.Fallback(),
                        )(relax_mod)

            ex = relax.build(relax_mod, target_str)
            compile_time = time.perf_counter()-t1
            print(net_name+" compile time:" + str(compile_time))
            ex.export_library(path)
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
        # 没有解决input list 保存的问题
        # else:
        #     ex = tvm.runtime.load_module(path)
        #     with open(param_path, "r") as file:
        #         input_list = json.load(file)

        vm = relax.VirtualMachine(ex, dev)

        # Setup execution
        ftimer = vm.time_evaluator("main", dev)
        # perf = ftimer(*input_list).mean
        # print(perf)
        mean_perf, std_perf = measure(ftimer, target_str, *input_list)
        return mean_perf, compile_time, ''


def measure_end_to_end_tvm_afusion(stage, net, params, target_str, shape_dict, method_mode, net_name, hw_name, batch_size):
    with tvm.transform.PassContext(opt_level=OPT_LEVEL):
        if (stage >= 3):
            net = collage(net)
        relax_mod = relay_translator.from_relay(
            net, target=target_str, relay_params=params)
        dev = tvm.device(target_str, 0)
        if (target_str == "cuda"):
            seq = tvm.transform.Sequential(
                [relax.transform.LegalizeOps(), tvm.tir.transform.DefaultGPUSchedule()]
            )
            relax_mod = seq(relax_mod)

        if (stage >= 1):
            seq = tvm.transform.Sequential(
                [
                    relax.transform.DecomposeOpsForInference(),
                    relax.transform.LegalizeOps(),
                    relax.transform.AnnotateTIROpPattern(),
                    relax.transform.FuseOps(),
                    relax.transform.FuseTIR()
                ]
            )
            relax_mod = seq(relax_mod)

        if (stage == 2 or stage >= 4):
            with args.target:
                seq = tvm.transform.Sequential(
                    [
                        dl.ApplyDefaultSchedule(
                            # dl.gpu.Matmul(),
                            dl.gpu.Transpose(),
                            dl.gpu.Reduction(),
                            # dl.gpu.Transpose(),
                            dl.gpu.DecodeGEMV(),
                            dl.gpu.Matmul(),
                            dl.gpu.Fallback(),
                        )
                    ]
                )
                relax_mod = seq(relax_mod)
        ex = relax.build(relax_mod, target_str)
        if (stage >= 5):

            # with target_str, PassContext(opt_level=3):
            #     tuning_pass = relax.transform.MetaScheduleTuneIRMod(
            #         params={}, work_dir=database_dir, max_trials_global=4, max_trials_per_task=8,
            #     )
            #     relax_mod = tuning_pass(relax_mod)

            #     application_pass = relax.transform.MetaScheduleApplyDatabase(database_dir)
            #     relax_mod = application_pass(relax_mod)

            database = ms.relax_integration.tune_relax(
                mod=relax_mod,
                target=target_str,
                params=params,
                work_dir=database_dir,
                # for faster tuning
                max_trials_global=20000,
                max_trials_per_task=8,
                num_trials_per_iter=8,
                strategy="replay-trace",
                # max_trials_global=20000,
                # num_trials_per_iter=32,
                # max_trials_per_task=128,
                # strategy="evolutionary",

                space=ms.space_generator.PostOrderApply(
                    sch_rules=sch_rules,
                    postprocs=postprocs,
                    mutator_probs={},
                ),
                # This enables anchor-block tuning, where different subgraphs
                # with the same anchor block workload will be identified as equal.
                # It reduces the number of conv2d tuning tasks in the int8 resnet50 model
                # from 36 to 23, with negligible performance difference.
                module_equality="anchor-block",
            )

            ex = ms.relay_integration.compile_relay(
                database=database,
                mod=mod,
                target=target_str,
                params=params,
            )
        # ex = relax.build(relax_mod, target_str)
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
    ftimer = vm.time_evaluator("main", dev)
    mean_perf, std_perf = measure(ftimer, target_str, *input_list)
    return mean_perf, std_perf, relax_mod


def measure_relay(mod, params, shape_dict, args, is_perf_logging):
    mean_perf, std_perf, relax_mod = measure_end_to_end_tvm_no_tuning(mod["main"], params, args.target, shape_dict,
                                                                      None,
                                                                      args.network, args.hw, args.batch_size)
    print(f"[{args.network}] Performance of relay on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_afusion_perf(args, 'relay', mean_perf, std_perf, is_perf_logging)


def measure_relax(mod, params, shape_dict, args, is_perf_logging, stage):
    seq = tvm.transform.Sequential(
        [
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            relay.transform.CanonicalizeOps(),
            relay.transform.AlterOpLayout(),
            relay.transform.FoldConstant(),
        ]
    )
    mod = seq(mod)
    mean_perf, compile_time, relax_mod = measure_relax_dlight(stage, mod["main"], params, args.target, shape_dict,
                                                              None,
                                                              args.network, args.hw, args.batch_size)
    print(f"[{args.network}] Performance of afusion {stage} on {args.hw} (mean, std) = ({mean_perf:.4f}+-{compile_time:.4f})")
    log_afusion_perf(args, "relax", mean_perf, compile_time, is_perf_logging)

    # verify_network_output(mod["main"], shape_dict, mod_tvm, mod_evo)


def measure_afusion(mod, params, shape_dict, args, is_perf_logging, stage):
    if (stage >= 1):
        seq = tvm.transform.Sequential(
            [
                relay.transform.SimplifyInference(),
                relay.transform.FoldConstant(),
                relay.transform.FoldScaleAxis(),
                relay.transform.CanonicalizeOps(),
                relay.transform.AlterOpLayout(),
                relay.transform.FoldConstant(),
            ]
        )
        mod = seq(mod)
    mean_perf, std_perf, relax_mod = measure_end_to_end_tvm_afusion(stage, mod["main"], params, args.target, shape_dict,
                                                                    None,
                                                                    args.network, args.hw, args.batch_size)
    print(f"[{args.network}] Performance of afusion {stage} on {args.hw} (mean, std) = ({mean_perf:.4f}+-{std_perf:.4f})")
    log_afusion_perf(args, stage, mean_perf, std_perf, is_perf_logging)

    # verify_network_output(mod["main"], shape_dict, mod_tvm, mod_evo)


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

    # 运行时间， tuning time， accuracy
    # stage 1~5
    args.target = get_build_target(args.hw)
    print(args.target)
    logging.basicConfig(level=logging.INFO)
    args.func = "model"
    mod, params, shape_dict, _ = get_network_from_torch(
        args.network, args.batch_size)
    is_perf_logging = True
    # measure_afusion(mod, params, shape_dict, args, is_perf_logging, 2)
    if (args.runner == "relax"):
        measure_relax(mod, params, shape_dict, args, is_perf_logging,  1)
    elif (args.runner == "dlight"):
        measure_relax(mod, params, shape_dict, args, is_perf_logging,  2)
    elif (args.runner == "relay"):
        pass


# if __name__ == "__main__":
#     network = "bert_full"
#     mod, params, shape_dict, _ = get_network_from_torch(network, 1)
#     measure_relax_dlight(1, mod["main"], params, "cuda", shape_dict, "", network, "cuda", 1)
