from utils import Split
from testing import Testing

import time, psutil


def evaluate(dataset, model, model_name, embed_f, limit_num_sents):
    split = Split(embed_f)

    # TRAINING
    start_time_train = time.time()

    # Split dataset
    X_train, y_train = split.get_X_y(dataset['train'], limit_num_sents=limit_num_sents, set_type='train')

    # Train
    model.fit(X_train, y_train)

    end_time_train = time.time()

    memory = psutil.Process().memory_full_info().uss / (1024 ** 2)  # in megabytes

    # TESTING
    start_time_inference = time.time()

    # Split dataset
    X_test, y_test = split.get_X_y(dataset['test'], limit_num_sents=None, set_type='test')

    if model_name == 'ADBThreshold':
        model.oos_label = split.intents_dct['oos']

    # Test
    testing = Testing(model, X_test, y_test, model_name, split.intents_dct['oos'])
    results_dct = testing.test_train()

    end_time_inference = time.time()

    results_dct['time_train'] = round(end_time_train - start_time_train, 1)
    results_dct['time_inference'] = round(end_time_inference - start_time_inference, 1)
    results_dct['memory'] = round(memory, 1)

    return results_dct
