from ml_collections import config_dict


def get_config():
  config = config_dict.ConfigDict(initial_dictionary=dict(
    save_dir='', data_root='',
  ))
  config.seed = 0
  config.domain = 'deepcoder'
  config.io_encoder = 'lambda_signature'
  config.model_type = 'deepcoder'
  config.value_encoder = 'lambda_signature'
  config.min_num_examples = 2
  config.max_num_examples = 5
  config.min_num_inputs = 1
  config.max_num_inputs = 3
  config.max_search_weight = 12
  config.beam_size = 10
  config.num_proc = 1
  config.gpu_list = '0'
  config.gpu = 0
  config.embed_dim = 64
  config.port = '29500'
  config.use_ur = True
  config.stochastic_beam = False
  config.do_test = True
  config.synthetic_test_tasks = False
  config.json_results_file = ''
  config.timeout = 60
  config.restarts_timeout = 0
  config.temperature = 1.0
  config.encode_weight = True
  config.train_data_glob = ''
  config.test_data_glob = ''
  config.random_beam = False
  config.use_op_specific_lstm = True
  config.load_model = ''
  config.data_name = ''
  return config
