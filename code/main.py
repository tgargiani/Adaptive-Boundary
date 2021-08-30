from utils import DS_CLINC150_PATH, USE_DAN_PATH, USE_TRAN_PATH, print_results, get_intents_selection, filter, \
    DS_ROSTD_PATH
from custom_embeddings import create_embed_f
from ADBThreshold import ADBThreshold
from ood_train import evaluate

import os, json, copy
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from statistics import mean

time_pretraining = None
accuracy_all_lst = []
f1_all_lst = []
f1_ood_lst = []
f1_id_lst = []
time_train_lst = []
time_inference_lst = []
memory_lst = []
time_pretraining_lst = []

# 0 - LOAD DATASET
# ------------------------------------------------------------
dataset_name = 'clinc150-data_full'
dataset_path = os.path.join(DS_CLINC150_PATH, 'data_full.json')
# ------------------------------------------------------------
# dataset_name = 'rostd_full'
# dataset_path = os.path.join(DS_ROSTD_PATH, 'rostd_full.json')
# ------------------------------------------------------------

with open(dataset_path) as f:
    old_dataset = json.load(f)

# 1 - LIMIT NUMBER OF SENTENCES?
LIMIT_NUM_SENTS = None  # either None (i.e. no limit) or int with value > 0 (i.e. maximal number of sentences per class).

# 2 - CHOOSE KNOWN RATIO
KNOWN_RATIO = 0.25

# 3 - CHOOSE MODEL & ALPHA
alpha = 0.35  # best for KNOWN_RATIO=0.25 (measured on CLINC150)
# alpha = 0.4  # best for KNOWN_RATIO=0.5 (measured on CLINC150)
# alpha = 0.75  # best for KNOWN_RATIO=0.75 (measured on CLINC150)
model = ADBThreshold(alpha=alpha)

model_name = type(model).__name__

# 4A - CHOOSE EMBEDDING (no pre-training)
embed_f = hub.load(USE_DAN_PATH)
emb_name = 'use_dan'

# embed_f = hub.load(USE_TRAN_PATH)
# emb_name = 'use_tran'
#
# embed_f = SentenceTransformer('stsb-roberta-base').encode
# emb_name = 'sbert'

# 4B-1 - CHOOSE EMBEDDING (with pre-training - uncomment also in the loop)
# use_dan = hub.load(USE_DAN_PATH)
# use_tran = hub.load(USE_TRAN_PATH)
# sbert = SentenceTransformer('stsb-roberta-base').encode

for r in range(10):
    dataset = copy.deepcopy(old_dataset)

    num_classes = len(set([x[1] for x in dataset['train']]))
    num_sel_classes = int(KNOWN_RATIO * num_classes)
    selection = get_intents_selection(dataset['train'], num_intents=num_sel_classes)

    filt_train = filter(dataset['train'], selection, 'train')
    filt_test = filter(dataset['test'], selection, 'test')

    dataset['train'] = filt_train
    dataset['test'] = filt_test

    # 4B-2 - CHOOSE EMBEDDING (with pre-training)
    # embed_f, time_pretraining = create_embed_f(use_dan, dataset, LIMIT_NUM_SENTS, type='cosface')
    # emb_name = 'use_dan_cosface'
    #
    # embed_f, time_pretraining = create_embed_f(use_dan, dataset, LIMIT_NUM_SENTS, type='triplet_loss')
    # emb_name = 'use_dan_triplet_loss'
    #
    # embed_f, time_pretraining = create_embed_f(use_tran, dataset, LIMIT_NUM_SENTS, type='cosface')
    # emb_name = 'use_tran_cosface'
    #
    # embed_f, time_pretraining = create_embed_f(use_tran, dataset, LIMIT_NUM_SENTS, type='triplet_loss')
    # emb_name = 'use_tran_triplet_loss'
    #
    # embed_f, time_pretraining = create_embed_f(sbert, dataset, LIMIT_NUM_SENTS, type='cosface')
    # emb_name = 'sbert_cosface'
    #
    # embed_f, time_pretraining = create_embed_f(sbert, dataset, LIMIT_NUM_SENTS, type='triplet_loss')
    # emb_name = 'sbert_triplet_loss'

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

if LIMIT_NUM_SENTS is not None:
    print(f'sentences limited to {LIMIT_NUM_SENTS}')

print(f'dataset: {dataset_name}, embedding: {emb_name}, known ratio: {KNOWN_RATIO}, alpha: {alpha}')
print(
    f'accuracy_all: {round(mean(accuracy_all_lst), 1)},'
    f' f1_all: {round(mean(f1_all_lst), 1)},'
    f' f1_ood: {round(mean(f1_ood_lst), 1)},'
    f' f1_id: {round(mean(f1_id_lst), 1)},'
    f' time_train: {round(mean(time_train_lst), 1)},'
    f' time_inference: {round(mean(time_inference_lst), 1)},'
    f' memory: {round(mean(memory_lst), 1)},'
    f' time_pretraining: {round(mean(time_pretraining_lst), 1) if len(time_pretraining_lst) != 0 else 0}')
