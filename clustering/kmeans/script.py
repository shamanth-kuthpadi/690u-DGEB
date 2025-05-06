import torch
import dgeb
from functools import partial
from datasets import load_dataset

ALL_DEVICES = list(range(torch.cuda.device_count()))
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEQ_LEN = 1024


get_model = partial(
    dgeb.get_model,
    devices=ALL_DEVICES,
    batch_size=DEFAULT_BATCH_SIZE,
    max_seq_length=DEFAULT_SEQ_LEN,
)
dataset = load_dataset("tattabio/e_coli_rnas") 
protein_tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
protein_evaluation = dgeb.DGEB(tasks=protein_tasks)
protein_evaluation.run(get_model("facebook/esm2_t6_8M_UR50D"))