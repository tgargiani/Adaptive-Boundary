from utils import print_results, get_intents_selection, filter, DS_CLINC150_PATH, DS_BANKING77_PATH
from custom_embeddings import create_embed_f
from ADBThreshold import ADBThreshold
from ood_train import evaluate

import os, json, copy
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from statistics import mean
import sys


def run(dataset_name):
    # LOAD DATASET
    if dataset_name == 'clinc150':
        dataset_path = os.path.join(DS_CLINC150_PATH, 'data_full.json')
    elif dataset_name == 'banking77':
        dataset_path = os.path.join(DS_BANKING77_PATH, 'banking77.json')

    with open(dataset_path) as f:
        old_dataset = json.load(f)

    # LIMIT NUMBER OF SENTENCES?
    LIMIT_NUM_SENTS = None  # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).

    if LIMIT_NUM_SENTS is not None:
        print(f'sentences limited to {LIMIT_NUM_SENTS}')

    for KNOWN_RATIO, alpha in zip([0.25, 0.50, 0.75], [0.35, 0.4, 0.75]):
        for test_idx in [1, 2, 3, 4, 5, 6, 7, 8, 9]:

            time_pretraining = None
            accuracy_all_lst = []
            f1_all_lst = []
            f1_ood_lst = []
            f1_id_lst = []
            time_train_lst = []
            time_inference_lst = []
            memory_lst = []
            time_pretraining_lst = []

            for r in range(10):
                model = ADBThreshold(alpha=alpha)
                model_name = type(model).__name__

                # CHOOSE EMBEDDING
                if test_idx in [1, 4, 5]:
                    embed_f = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                    emb_name = 'use_dan'
                elif test_idx in [2, 6, 7]:
                    embed_f = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
                    emb_name = 'use_tran'
                elif test_idx in [3, 8, 9]:
                    embed_f = SentenceTransformer('stsb-roberta-base').encode
                    emb_name = 'sbert'

                dataset = copy.deepcopy(old_dataset)

                num_classes = len(set([x[1] for x in dataset['train']]))
                num_sel_classes = int(KNOWN_RATIO * num_classes)
                selection = get_intents_selection(dataset['train'], num_intents=num_sel_classes)

                filt_train = filter(dataset['train'], selection, 'train')
                filt_test = filter(dataset['test'], selection, 'test')

                dataset['train'] = filt_train
                dataset['test'] = filt_test

                # CHOOSE PRE-TRAINING
                if test_idx == 4:
                    embed_f, time_pretraining = create_embed_f(embed_f, dataset, LIMIT_NUM_SENTS, type='cosface')
                    emb_name = 'use_dan_cosface'
                elif test_idx == 5:
                    embed_f, time_pretraining = create_embed_f(embed_f, dataset, LIMIT_NUM_SENTS,
                                                               type='triplet_loss')
                    emb_name = 'use_dan_triplet_loss'
                elif test_idx == 6:
                    embed_f, time_pretraining = create_embed_f(embed_f, dataset, LIMIT_NUM_SENTS, type='cosface')
                    emb_name = 'use_tran_cosface'
                elif test_idx == 7:
                    embed_f, time_pretraining = create_embed_f(embed_f, dataset, LIMIT_NUM_SENTS,
                                                               type='triplet_loss')
                    emb_name = 'use_tran_triplet_loss'
                elif test_idx == 8:
                    embed_f, time_pretraining = create_embed_f(embed_f, dataset, LIMIT_NUM_SENTS, type='cosface')
                    emb_name = 'sbert_cosface'
                elif test_idx == 9:
                    embed_f, time_pretraining = create_embed_f(embed_f, dataset, LIMIT_NUM_SENTS,
                                                               type='triplet_loss')
                    emb_name = 'sbert_triplet_loss'

                results_dct = evaluate(dataset, model, model_name, embed_f, LIMIT_NUM_SENTS)

                accuracy_all_lst.append(results_dct['accuracy_all'])
                f1_all_lst.append(results_dct['f1_all'])
                f1_ood_lst.append(results_dct['f1_ood'])
                f1_id_lst.append(results_dct['f1_id'])
                time_train_lst.append(results_dct['time_train'])
                time_inference_lst.append(results_dct['time_inference'])
                memory_lst.append(results_dct['memory'])

                if time_pretraining is not None:
                    time_pretraining_lst.append(time_pretraining)

                print_results(dataset_name, model_name, emb_name, results_dct)

            accuracy_all = round(mean(accuracy_all_lst), 1)
            f1_all = round(mean(f1_all_lst), 1)
            f1_ood = round(mean(f1_ood_lst), 1)
            f1_id = round(mean(f1_id_lst), 1)
            time_train = round(mean(time_train_lst), 1)
            time_inference = round(mean(time_inference_lst), 1)
            memory = round(mean(memory_lst), 1)
            time_pretraining = round(mean(time_pretraining_lst), 1) if len(time_pretraining_lst) != 0 else 0

            print(
                f'test_idx: {test_idx}, dataset: {dataset_name}, embedding: {emb_name},'
                f' known ratio: {KNOWN_RATIO}, alpha: {alpha}')
            print(
                f'accuracy_all: {accuracy_all},'
                f' f1_all: {f1_all},'
                f' f1_ood: {f1_ood},'
                f' f1_id: {f1_id},'
                f' time_train: {time_train},'
                f' time_inference: {time_inference},'
                f' memory: {memory},'
                f' time_pretraining: {time_pretraining}')

            with open(f'{dataset_name}_results.txt', 'a') as f:
                f.write(
                    f'test_idx: {test_idx},'
                    f' dataset: {dataset_name},'
                    f' embedding: {emb_name},'
                    f' known ratio: {KNOWN_RATIO},'
                    f' alpha: {alpha},'
                    f' accuracy_all: {accuracy_all},'
                    f' f1_all: {f1_all},'
                    f' f1_ood: {f1_ood},'
                    f' f1_id: {f1_id},'
                    f' time_train: {time_train},'
                    f' time_inference: {time_inference},'
                    f' memory: {memory},'
                    f' time_pretraining: {time_pretraining}'
                    f'\n'
                )


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in ['clinc150', 'banking77']:
        run(sys.argv[1])
    else:
        print('Usage: python3 run.py {clinc150,banking77}')
