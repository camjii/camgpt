class BytePairEncoding():
    def __init__(self, text):
        self.text = text
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
    

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        for pair, new_id in self.merges.items():
            ids = self.merge_pairs(new_id, pair, ids)
    
        return ids
    
    def count_pairs(self, ids):
        pairs = {}
        for pair in zip(ids, ids[1:]):
            pairs[pair] = pairs.get(pair, 0) +1 #seeing if pair appears in pairs already

        
        max_pair = max(pairs, key=pairs.get)
        
        return pairs, max_pair
    

    def merge_pairs(self, new_id, max_pair, ids):
        new_ids = []

        i = 0
        while i < len(ids):
            if  i < len(ids) - 1 and max_pair ==(ids[i], ids[i+1]):
                new_ids.append(new_id)
                i+=2
            
            else:
                new_ids.append(ids[i])
                i+=1
        
        if new_id in new_ids:
            self.vocab[new_id] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
        return new_ids
    

    def decode(self, new_ids):
       decoded_str = b''.join([self.vocab[i] for i in new_ids])

       return decoded_str.decode('utf-8', errors='replace')

        


    def train(self, vocab_size):
        ids = list(self.text.encode('utf-8'))
        for new_id in range(256, vocab_size):
            if len(ids) < 2:
                break
            pairs, max_pair = self.count_pairs(ids)
            self.merges[max_pair] = new_id
            ids = self.merge_pairs(new_id, max_pair, ids)




