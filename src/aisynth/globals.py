import torch

TEST_INPUTS = {"AUDIO_DIR_TEST": [{"audio_dir": r"test_resources/dx7"}],
               "CONVBLOCK_TEST": [{"n_in": 10,
                                   "n_hidden": 10,
                                   "batchnorm_flag": False,
                                   "dropout_flag": False,
                                   "test_tensor": torch.rand(10, 10)},
                                  {"n_in": 10,
                                   "n_hidden": 10,
                                   "batchnorm_flag": True,
                                   "dropout_flag": True,
                                   "test_tensor": torch.rand(10, 10)}
                                  ],
               "RESBLOCK_TEST": [{"n_in": 10,
                                  "n_depth": 2,
                                  "dilation_growth_rate": 2,
                                  "dilation_cycle": None,
                                  "reverse_dilation": False,
                                  "m_conv": 1,
                                  "test_tensor": torch.rand(10, 10)},
                                 {"n_in": 10,
                                  "n_depth": 6,
                                  "dilation_growth_rate": 4,
                                  "dilation_cycle": None,
                                  "reverse_dilation": True,
                                  "m_conv": 2,
                                  "test_tensor": torch.rand(10, 10)}
                                 ]

               }
