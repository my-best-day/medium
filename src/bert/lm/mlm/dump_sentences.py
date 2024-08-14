import torch
"""
This module contains a class that converts tokenized predictions and labels to human readable
format.
"""


class DumpStentences:
    """
    Takes a tokonized prediction and the corresponding labels and converts them to a human readable
    format.

    :param tokenizer: The tokenizer used to convert the token ids to strings.
    """
    def __init__(self, tokenizer):
        """
        Takes a tokonized prediction and the corresponding labels and converts them to a human
        readable format.
        """
        self.tokenizer = tokenizer

    def batched_debug(self, sentence, labels, mlm_out):
        """
        Converts a batch of tokenized predictions and the corresponding labels to human readable
        format.

        :param sentence: The tokenized input sentence.
        :param labels: The tokenized labels.
        :param mlm_out: The tokenized prediction.
        :return: A list of pair of human readable sentences - the prediction and the original
        sentence.
        """
        with torch.no_grad():
            B, _, _ = mlm_out.shape
            print(f"Batch size: {B}")
            text = []
            for b in range(B):
                if any(element != 0 for element in labels[b]):
                    english = self.debug(sentence[b], labels[b], mlm_out[b])
                    text.append(english)
                    if len(text) >= 30:
                        break
                else:
                    print("skipping")
            return text

    def debug(self, sentence, labels, mlm_out):
        """
        Converts a tokenized prediction and the corresponding labels to human readable format.

        For example, returns:

        "The fox /jumped/ over the /red/ dog /./"
        "\n"
        "The fox jumped over the lazy dog."

        :param sentence: The tokenized input sentence.
        :param labels: The tokenized labels.
        :param mlm_out: The tokenized prediction.
        :return: A pair of human readable sentences - the prediction and the original sentence
        """
        english = []
        for i, id in enumerate(labels):
            if id != 0:
                predicted_id = mlm_out[i].argmax(axis=-1)
                token = f"/{self.convert_id_to_token(predicted_id)}/"
                # print(f"{predicted_id} -> {token} : {describe(mlm_out[i])}")
            else:
                token = self.convert_id_to_token(sentence[i])
            english.append(token)
        english = self.convert_tokens_to_string(english)
        sentence2 = map(lambda i: labels[i] if sentence[i] == self.tokenizer.mask_token_id else
                        sentence[i], range(len(sentence)))
        source = self.tokenizer.convert_ids_to_tokens(sentence2)
        source = self.convert_tokens_to_string(source)
        english = english[:144]
        source = source[:144]
        return f"{english}\n{source}"

    def convert_id_to_token(self, id):
        token = self.tokenizer.convert_ids_to_tokens([id])[0]
        return token

    def convert_tokens_to_string(self, tokens):
        import re
        text = self.tokenizer.convert_tokens_to_string(tokens)
        # remove the [CLS] and [SEP] tokens
        cleaned_text = re.sub(r'\[.*?\]\s*', '', text)
        return cleaned_text


def describe_tensor(tensor):
    """
    Returns descriptive statistics for a 1D PyTorch tensor.

    :param tensor: A 1D PyTorch tensor.
    :return: A dictionary with min, max, mean, standard deviation, 25th percentile, and 75th
        percentile.
    """
    if tensor.dim() != 1:
        raise ValueError("Tensor must be 1-dimensional.")

    desc_stats = {
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "mean": tensor.mean().item(),
        "stddev": tensor.std().item(),
        "25%": torch.quantile(tensor, 0.25).item(),
        "75%": torch.quantile(tensor, 0.75).item()
    }

    return desc_stats


def describe(tensor):
    """
    print the descriptive statistics for a 1D PyTorch tensor as one line of text.
    """
    stats = describe_tensor(tensor)
    print(" | ".join([f"{k}: {v:.2f}" for k, v in stats.items()]))
