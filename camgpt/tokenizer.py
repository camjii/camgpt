class Tokenizer():
    def __init__(self, text, vocab, num_merges, merged):
        self.text = text
        self.vocab = vocab
        self.num_merges = num_merges
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
                self.merged[new_id] = pair   #Adding the new merge to the dictionary
                i+=2 #move to next pair 
            else:
                new_ids.append(ids[i])
                i+=1
        
        if i == len(ids) - 1: #Ensuring last position does not get skipped
            new_ids.append(ids[i])

        return new_ids
    

    def expand(self, id):  #Helper function to recursively deal with tokens that have been merged multiple times
        if id not in self.merged.keys():
            return [id]
        else:
            pair = self.merged[id] #token is merged 
        return self.expand(pair[0]) + self.expand(pair[1]) #joining the two lists together to get raw values


    def decode(self, ids):
        decoding= []
        for id in ids:
            decoding.extend(self.expand(id))

        return bytes(decoding).decode('utf-8') #turning integers back into strings
    
    
    def train(self):
        ids = self.encode()
        
        counted_pairs = self.count_pairs(ids)

        for i in range(self.num_merges):
            max_pair = max(counted_pairs,key = counted_pairs.get) #gets the max
            new_id = len(self.vocab) +i
            merged = self.merge(ids,max_pair,new_id)
            ids = merged #updating ids to represent merged tokens
            counted_pairs = self.count_pairs(ids) #recounting pairs for further merging

        return ids
        

                  
    
   
   
