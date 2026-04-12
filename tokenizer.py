
class Tokenizer():
    def __init__(self, text, vocab_size=276):
        self.text = text
        self.vocab_size = vocab_size
        self.merges = {}

        self.vocab = {i: bytes([i]) for i in range(256)}

    def encode(self):
        ids = list(self.text.encode('utf-8')) #list of ids
        return ids

    def _get_pair_counts(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]): #zip acts as a pairing mechanism by pairing each element in ids with every next element
            counts[pair] = counts.get(pair, 0) + 1 #adds one to either pair or 0 if not seen
        return counts

    def _merge(self, ids, pair, new_id):
        """Replace all occurrences of `pair` in `ids` with `new_id`.

        For example: ids=[1,2,3,1,2], pair=(1,2), new_id=99
        should return [99,3,99]

        Args:
            ids: list of token ids
            pair: tuple (id1, id2) to replace
            new_id: the replacement token id

        Returns:
            new list with the pair merged
        """
        new_ids = []

        i = 0
        while i<len(ids):
            if i < len(ids) -1 and pair == (ids[i], ids[i+1]): #we use len(ids) - 1 because we are checking pairwise so if we were to do len(ids) plainly it would be out of bounds
                new_ids.append(new_id)
                i+=2 #incrementing to next pair
            else:
                new_ids.append(ids[i])
                i +=1
        return new_ids
    
    def train(self):
        """Learn BPE merges from self.text until vocab reaches vocab_size.

        This is the training loop. It should:
        1. Start with the raw UTF-8 byte ids
        2. Repeatedly find the most frequent pair
        3. Merge that pair into a new token
        4. Record the merge and update the vocab
        5. Stop when vocab_size is reached (or no pairs remain)

        Populates self.merges and self.vocab.
        """
        ids = self.encode()
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            
            pair_counts = self._get_pair_counts(ids)
            if not pair_counts:
                break 
            maxpair = max(pair_counts, key=pair_counts.get)

            newtok = 256 +i
            ids = self._merge(ids, maxpair, newtok)

            self.merges[maxpair] = newtok

            self.vocab[newtok] = self.vocab[maxpair[0]] + self.vocab[maxpair[1]] #encoding the newtok in the vocab



        return ids


    def tokenize(self, text):
        ids = list(text.encode('utf-8'))


        for pair, new_id in self.merges.items():
            ids = self._merge(ids,pair,new_id)
            return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[id] for id in ids)
        return tokens.decode('utf-8', errors='replace')



if __name__ == "__main__":
    sample = "TESTtest"
    tok = Tokenizer(sample, vocab_size=260)

    #Train
    tok.train()
    print(f"Learned merges: {tok.merges}")

    #Encode
    encoded = tok.tokenize(sample)
    print(f"Encoded: {encoded}")

    # Decode
    decoded = tok.decode(encoded)
    print(f"Decoded: {decoded}")

    assert decoded == sample, "Round-trip failed!"
    print("Round-trip OK!")
