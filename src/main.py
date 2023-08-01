#python 3.10

#requirements
# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# pip install git+https://git@github.com/monologg/KoBERT-Transformers.git@master
#pip install sentencepiece
#pip install transformers
#pip install torch

from models import data_loader, model_builder, trainer
from kobert_transformers.tokenization_kobert import KoBertTokenizer

from pre_process import data_parser
import pandas as pd
from sklearn.model_selection import train_test_split as tts
import torch
import argparse
import json
import os.path
import sys

def add_argument(file_loc):
    parser = argparse.ArgumentParser()
    file_loc = file_loc.replace('/', '\\')

    try:
        config = json.load(open(sys.path[0] + '/'+ file_loc, 'r'))
        t_args = argparse.Namespace()
        t_args.__dict__.update(config)
        args = parser.parse_args(namespace=t_args)

        return args

    except Exception as e:
        print(e)

        parser.add_argument("-dir_name", default='../data/')
        parser.add_argument("-chk_point", default='../models/')

        parser.add_argument("-tokenizer_len", default=512)

        parser.add_argument("-encoder", default='classifier', type=str,
                            choices=['classifier', 'rnn'])
        parser.add_argument("-mode", default='train', type=str, choices=['train', 'summary'])

        parser.add_argument("-input_size", default=768)
        parser.add_argument("-hidden_size", default=768)
        parser.add_argument("-num_layer", default=2)

        parser.add_argument("-drop_rate_encoder", default=0.5)
        parser.add_argument("-drop_rate_bert", default=0.5)

        parser.add_argument("-batch_size", default=8)
        parser.add_argument("-warmup_rate", default=1)
        parser.add_argument("-epoch", default=10)
        parser.add_argument("-grad_norm", default=1)
        parser.add_argument("-learn_rate", default=1)


if __name__ == '__main__':
    argument = add_argument('../data/config.json')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_builder.Classifier(argument).to(device)

    if(argument.mode == "train"):
        epoch = argument.epoch

        # df = pd.read_excel(argument.dir_name + argument.train_file)
        dset_train = pd.read_table(argument.dir_name + argument.train_file)
        dset_test = pd.read_table(argument.dir_name + argument.test_file)

        dset_train = data_loader.ElectraDataset(dset_train['document'], dset_train['label'], argument)
        dset_test = data_loader.ElectraDataset(dset_test['document'], dset_test['label'], argument)

        dset_train = torch.utils.data.DataLoader(dset_train, batch_size=argument.batch_size, num_workers=1)
        dset_test = torch.utils.data.DataLoader(dset_test, batch_size=argument.batch_size, num_workers=1)

        # argument['dset_train'] = dset_train
        # argument['dset_test'] = dset_test

        trainer = trainer.Trainer(model,device,dset_train, dset_test, argument)

        for e in range(epoch):
            trainer.train_loop()
            trainer.test_loop()
            torch.save(model, argument.dir_name + argument.chk_point)

    elif(argument.mode == "validate"):

        df = pd.read_table(argument.dir_name + argument.valid_file)

        dset = data_loader.ValidDataset(df['document'])
        dset = torch.utils.data.DataLoader(dset, batch_size=argument.batch_size, num_workers=1)

        trainer = trainer.Trainer(model, device, argument)

        result = trainer.Trainer.valid_loop(dset)

        df['result'] = result

        df.to_excel(argument.dir_name + 'result.xlsx')

