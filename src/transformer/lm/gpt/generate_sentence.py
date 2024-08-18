import torch


class GenerateSentence:
    def __init__(self, model, tokenizer, seq_len, max_new_tokens):
        assert 0 < max_new_tokens < seq_len, f"max_new_tokens must be between 0 and {seq_len}"

        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_new_tokens = max_new_tokens

    def batched_debug(self, encoded_prompt, *_):
        """
        Extends the input prompt with new tokens and returns the generated sentences

        Assuming @no_gard and model.eval() are called before this function
        """
        # only take up to 3 samples from encoded_prompt
        encoded_prompt = encoded_prompt[:3]
        encoded_response = self.generate(encoded_prompt)
        # convert to human readable format
        # Create responses with a divider between original and generated text
        responses = []
        for i in range(len(encoded_prompt)):
            original = self.tokenizer.decode(encoded_prompt[i], skip_special_tokens=True)
            generated = self.tokenizer.decode(encoded_response[i][len(encoded_prompt[i]):],
                                              skip_special_tokens=True)
            # Add divider between the original and generated text
            responses.append(f"{original} <| *** generated *** |> {generated}")

        return responses

    def generate(self, encoded_prompt):
        """
        Generate new tokens from a prompt

        Assuming @no_gard and model.eval() are called before this function

        :param prompt: The tokenized input sentence.

        """
        idx = encoded_prompt
        for _ in range(self.max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.seq_len:]
            # get the predictions
            logits = self.model(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
