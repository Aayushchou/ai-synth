import torch

TEST_INPUTS = {"AUDIO_DIR_TEST": [{"audio_dir": r"test_resources/dx7"}],
               "1DRESBLOCK_TEST": [{"n_in": 10,
                                    "n_hidden": 10,
                                    "batchnorm_flag": False,
                                    "dropout_flag": False,
                                    "test_tensor": torch.rand(10,10)},
                                   {"n_in": 10,
                                   "n_hidden": 10,
                                    "batchnorm_flag": True,
                                    "dropout_flag": True,
                                    "test_tensor": torch.rand(10,10)}]}

