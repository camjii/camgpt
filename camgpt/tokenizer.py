


class Tokenizer():
    def __init__(self, text, vocab, merged):
        self.text = text
        self.vocab = vocab
        self.merged = merged
        pass


    def encode(self):
        ids = list(self.text.encode('utf-8'))
        return ids
    
    def count_pairs(self,ids):
        pairs = {}
        for pair in zip(ids, ids[1:]):
            pairs[pair] = pairs.get(pair, 0) +1 #get current count else default to 0

        return pairs

    def merge(self,ids, pair,new_id):
         #Merge pairs into a single token
         #1. Identify pairs and new token ids
         #2. Merge it into a single token id whilst encoding meaning
         #3. Add new token to vocab
       
        i = 0
        new_ids = []
        
        while i<len(ids) -1: #Since we are incrementing by 2
            
            if pair == (ids[i], ids[i+1]): #Checking if pair is in ids
                new_ids.append(new_id)
                self.merged[new_id] = pair
                i+=2 #move to next pair 
            else:
                new_ids.append(ids[i])
                i+=1
        
        if i == len(ids) - 1: #Ensuring last position does not get skipped
            new_ids.append(ids[i])

        return new_ids
    
    def train(self, text):
        ids = self.encode(text)
        
        counted_pairs = self.count_pairs(ids)
        

        #Code the rest out

                  
    
   
   
