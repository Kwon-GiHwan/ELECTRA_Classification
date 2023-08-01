from transformers import ElectraTokenizer
from torch.utils.data import Dataset

class ElectraDataset(Dataset):
    def __init__(self, sentence, label, arguments=None):

        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.sentence = sentence
        self.label = label
        self.max_length = arguments.max_length

    def __getitem__(self, idx):
        input = self.tokenizer(
            self.sentence[idx],
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True
        )


        return (input['input_ids'][0], input['attention_mask'][0], self.label[idx])

    def __len__(self):
        return (len(self.sentence))

class ValidDataset(Dataset):
    def __init__(self, sentence):
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.sentence = sentence

    def __getitem__(self, idx):
        input, attention = self.tokenizer(
            self.sentence[idx],
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True
        )

        return (input, attention)

    def __len__(self):
        return (len(self.sentence))



