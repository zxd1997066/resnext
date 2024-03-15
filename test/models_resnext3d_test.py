#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import unittest

import torch
from classy_vision.models import build_model, ClassyModel
#from test.generic.utils import compare_model_state

from generic.utils import compare_model_state
import time
import os

class TestResNeXt3D(unittest.TestCase):
    def setUp(self):
        model_config_template = {
            "name": "resnext3d",
            "input_key": "video",
            "clip_crop_size": 112,
            "skip_transformation_type": "postactivated_shortcut",
            "frames_per_clip": 32,
            "input_planes": 3,
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 3,
            "stage_planes": 64,
            "num_groups": 1,
            "width_per_group": 16,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "in_plane": 512,
                    "pool_size": (2, 7, 7),
                    "activation_func": "softmax",
                    "num_classes": 2,
                }
            ],
        }
        pbt = "postactivated_bottleneck_transformation"
        model_config_variants = [
            # # ResNeXt3D-34
            # {
            #     "residual_transformation_type": "basic_transformation",
            #     "num_blocks": [3, 4, 6, 3],
            # },
            # ResNeXt3D-50
            {"residual_transformation_type": pbt, "num_blocks": [3, 4, 6, 3]},
            # # ResNeXt3D-101
            # {"residual_transformation_type": pbt, "num_blocks": [3, 4, 23, 3]},
        ]

        self.model_configs = []
        for variant in model_config_variants:
            model_config = copy.deepcopy(model_config_template)
            model_config.update(variant)

            block_idx = model_config["num_blocks"][-1]
            # attach the head at the last block
            model_config["heads"][0]["fork_block"] = "pathway0-stage4-block%d" % (
                block_idx - 1
            )

            self.model_configs.append(model_config)

        self.batchsize = args.batch_size

        test_input = {"video": torch.rand(self.batchsize, 3, 16, 256, 320)}

        self.forward_pass_configs = {
            # "train": {
            #     # input shape: N x C x T x H x W
            #     "input": {"video": torch.rand(self.batchsize, 3, 16, 112, 112)},
            #     "model": {
            #         "stem_maxpool": False,
            #         "stage_temporal_stride": [1, 2, 2, 2],
            #         "stage_spatial_stride": [1, 2, 2, 2],
            #     },
            # },
            "test": {
                #"input": {"video": torch.rand(self.batchsize, 3, 16, 256, 320)},
                "input": test_input,
                "model": {
                    "stem_maxpool": True,
                    "stage_temporal_stride": [1, 2, 2, 2],
                    "stage_spatial_stride": [1, 2, 2, 2],
                },
            },
        }

    def test_build_model(self):
        for model_config in self.model_configs:
            model = build_model(model_config)
            self.assertTrue(isinstance(model, ClassyModel))

    def test_forward_pass(self):
        for split, split_config in self.forward_pass_configs.items():
            for model_config in self.model_configs:
                forward_pass_model_config = copy.deepcopy(model_config)
                forward_pass_model_config.update(split_config["model"])

                num_classes = forward_pass_model_config["heads"][0]["num_classes"]

                model = build_model(forward_pass_model_config)

                model = model.eval()
                model = model.cuda() if torch.cuda.is_available() else model
                if args.channels_last:
                    try:
                        model = model.to(memory_format=torch.channels_last_3d)
                        split_config["input"] = {k:v.to(memory_format=torch.channels_last_3d) 
                                for k,v in split_config["input"].items()}
                        print("---- Use channels last format.")
                    except:
                        print("---- Use normal format.")
                if args.compile:
                    model = torch.compile(model, backend=args.backend, options={"freezing": True})
                if args.ipex:
                    model.eval()
                    import intel_extension_for_pytorch as ipex
                    if args.precision == "bfloat16":
                        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
                    else:
                        model = ipex.optimize(model, dtype=torch.float32, inplace=True)
                    print("Running IPEX ...")
                if args.jit:
                    try:
                        # model = torch.jit.script(model)
                        split_config["input"]["video"] = split_config["input"]["video"].cuda() if torch.cuda.is_available() else split_config["input"]["video"]
                        model = torch.jit.trace(model, split_config["input"])
                        split_config["input"]["video"] = split_config["input"]["video"].cpu() if torch.cuda.is_available() else split_config["input"]["video"]
                        print("---- With JIT enabled.")
                        if args.ipex:
                            model = torch.jit.freeze(model)
                    except:
                        print("---- With JIT disabled.")

                warmup_steps = args.warmup_iters
                iters = args.num_iters
                # warmup
                for i in range(warmup_steps):
                    split_config["input"]["video"] = split_config["input"]["video"].cuda() if torch.cuda.is_available() else split_config["input"]["video"]
                    out = model(split_config["input"])
                    split_config["input"]["video"] = split_config["input"]["video"].cpu() if torch.cuda.is_available() else split_config["input"]["video"]
                ##run inference
                total_time = 0.0
                reps_done = 0
                batch_time_list = []
                for i in range(iters):
                    start = time.time()

                    split_config["input"]["video"] = split_config["input"]["video"].cuda() if torch.cuda.is_available() else split_config["input"]["video"]

                    if args.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                            out = model(split_config["input"])
                        #
                        if i == int(iters/2):
                            import pathlib
                            timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                            if not os.path.exists(timeline_dir):
                                os.makedirs(timeline_dir)
                            timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                        args.arch + str(i) + '-' + str(os.getpid()) + '.json'
                            print(timeline_file)
                            prof.export_chrome_trace(timeline_file)
                            table_res = prof.key_averages().table(sort_by="cpu_time_total")
                            print(table_res)
                            # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
                    else:
                        out = model(split_config["input"])
                    split_config["input"]["video"] = split_config["input"]["video"].cpu() if torch.cuda.is_available() else split_config["input"]["video"]
                    end = time.time()
                    print("Iteration: {}, inference time: {} sec.".format(i, end - start), flush=True)
                    delta = end - start
                    batch_time_list.append((end - start) * 1000)
                    total_time += delta
                    reps_done += 1
                avg_time = total_time / reps_done
                latency = avg_time * 1000
                throughput = 1.0 / avg_time
                # self.assertEqual(out.size(), (self.batchsize, num_classes))
                print("\n", "-"*20, "Summary", "-"*20)
                print("inference latency:\t {:.3f} ms".format(latency))
                print("inference Throughput:\t {:.2f} samples/s".format(throughput))
                # P50
                batch_time_list.sort()
                p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
                p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
                p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
                print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
                        % (p50_latency, p90_latency, p99_latency))


    def test_set_classy_state_plain(self):
        # We use the same model architecture to save and load a model state.
        # This is a plain use case of `set_classy_state` method
        for model_config in self.model_configs:
            model = build_model(model_config)
            model_state = model.get_classy_state()

            model2 = build_model(model_config)
            model2.set_classy_state(model_state)
            model2_state = model2.get_classy_state()
            compare_model_state(self, model_state, model2_state)

    def _get_model_config_weight_inflation(self):
        model_2d_config = {
            "name": "resnext3d",
            "frames_per_clip": 1,
            "input_planes": 3,
            "clip_crop_size": 224,
            "skip_transformation_type": "postactivated_shortcut",
            "residual_transformation_type": "postactivated_bottleneck_transformation",
            "num_blocks": [3, 4, 6, 3],
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 1,
            "stem_spatial_kernel": 7,
            "stem_maxpool": True,
            "stage_planes": 256,
            "stage_temporal_kernel_basis": [[1], [1], [1], [1]],
            "temporal_conv_1x1": [True, True, True, True],
            "stage_temporal_stride": [1, 1, 1, 1],
            "stage_spatial_stride": [1, 2, 2, 2],
            "num_groups": 1,
            "width_per_group": 64,
            "num_classes": 1000,
            "zero_init_residual_transform": True,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "pool_size": [1, 7, 7],
                    "activation_func": "softmax",
                    "num_classes": 1000,
                    "fork_block": "pathway0-stage4-block2",
                    "in_plane": 2048,
                    "use_dropout": False,
                }
            ],
        }

        model_3d_config = {
            "name": "resnext3d",
            "frames_per_clip": 8,
            "input_planes": 3,
            "clip_crop_size": 224,
            "skip_transformation_type": "postactivated_shortcut",
            "residual_transformation_type": "postactivated_bottleneck_transformation",
            "num_blocks": [3, 4, 6, 3],
            "input_key": "video",
            "stem_name": "resnext3d_stem",
            "stem_planes": 64,
            "stem_temporal_kernel": 5,
            "stem_spatial_kernel": 7,
            "stem_maxpool": True,
            "stage_planes": 256,
            "stage_temporal_kernel_basis": [[3], [3, 1], [3, 1], [1, 3]],
            "temporal_conv_1x1": [True, True, True, True],
            "stage_temporal_stride": [1, 1, 1, 1],
            "stage_spatial_stride": [1, 2, 2, 2],
            "num_groups": 1,
            "width_per_group": 64,
            "num_classes": 1000,
            "freeze_trunk": False,
            "zero_init_residual_transform": True,
            "heads": [
                {
                    "name": "fully_convolutional_linear",
                    "unique_id": "default_head",
                    "pool_size": [8, 7, 7],
                    "activation_func": "softmax",
                    "num_classes": 1000,
                    "fork_block": "pathway0-stage4-block2",
                    "in_plane": 2048,
                    "use_dropout": True,
                }
            ],
        }
        return model_2d_config, model_3d_config

    def test_set_classy_state_weight_inflation(self):
        # Get model state from a 2D ResNet model, inflate the 2D conv weights,
        # and use them to initialize 3D conv weights. This is an advanced use of
        # `set_classy_state` method.
        model_2d_config, model_3d_config = self._get_model_config_weight_inflation()
        model_2d = build_model(model_2d_config)
        model_2d_state = model_2d.get_classy_state()

        model_3d = build_model(model_3d_config)
        model_3d.set_classy_state(model_2d_state)
        model_3d_state = model_3d.get_classy_state()

        for name, weight_2d in model_2d_state["model"]["trunk"].items():
            weight_3d = model_3d_state["model"]["trunk"][name]
            if weight_2d.dim() == 5:
                # inflation only applies to conv weights
                self.assertEqual(weight_3d.dim(), 5)
                if weight_2d.shape[2] == 1 and weight_3d.shape[2] > 1:
                    weight_2d_inflated = (
                        weight_2d.repeat(1, 1, weight_3d.shape[2], 1, 1)
                        / weight_3d.shape[2]
                    )
                    self.assertTrue(torch.equal(weight_3d, weight_2d_inflated))

    def test_set_classy_state_weight_inflation_inconsistent_kernel_size(self):
        # Get model state from a 2D ResNet model, inflate the 2D conv weights,
        # and use them to initialize 3D conv weights.
        model_2d_config, model_3d_config = self._get_model_config_weight_inflation()
        # Modify conv kernel size in the stem layer of 2D model to 5, which is
        # inconsistent with the kernel size 7 used in 3D model.
        model_2d_config["stem_spatial_kernel"] = 5
        model_2d = build_model(model_2d_config)
        model_2d_state = model_2d.get_classy_state()
        model_3d = build_model(model_3d_config)
        with self.assertRaises(AssertionError):
            model_3d.set_classy_state(model_2d_state)
    
    def save_profile_result(filename, table):
        import xlsxwriter
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet()
        keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
                "CPU time avg", "Number of Calls"]
        for j in range(len(keys)):
            worksheet.write(0, j, keys[j])

        lines = table.split("\n")
        for i in range(3, len(lines)-4):
            words = lines[i].split(" ")
            j = 0
            for word in words:
                if not word == "":
                    worksheet.write(i-2, j, word)
                    j += 1
        workbook.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('-a', '--arch', type=str, default='ResNext3D',
            help='model architecture (default: resnet18)')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
            help='batch size')
    parser.add_argument('--ipex', action="store_true",
            help='ipex')
    parser.add_argument('--jit', action="store_true",
            help='jit')
    parser.add_argument('--profile', action="store_true",
            help='profile')
    parser.add_argument('-w', '--warmup_iters', type=int, default=10,
            help='warmup')
    parser.add_argument('-i', '--num_iters', type=int, default=100,
            help='iterations')
    parser.add_argument('--channels_last', type=int, default=0,
            help='NHWC')
    parser.add_argument('--precision', type=str, default='float32',
            help='float32, bfloat16')
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
    args = parser.parse_args()

    case = TestResNeXt3D()
    case.setUp()
    with torch.no_grad():
        if args.precision == 'bfloat16':
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
                case.test_forward_pass()
        elif args.precision == 'float16':
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
                case.test_forward_pass()
        else:
            case.test_forward_pass()
