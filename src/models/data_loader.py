from transformers import ElectraTokenizer
from torch.utils.data import Dataset

class ElectraDataset(Dataset):
    def __init__(self, sentence, label):

        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.sentence = sentence
        self.label = label

    def __getitem__(self, idx):
        input, attention = self.tokenizer(
            self.sentence[idx],
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            add_special_tokens=True
        )

        return (input, attention + (self.label[idx], ))

    def __len__(self):
        return (len(self.sentence))

class ElectraDataset_Validate(Dataset):
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



