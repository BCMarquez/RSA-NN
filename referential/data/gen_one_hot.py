import time
import json
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import os
   
import gensim
from gensim.models import KeyedVectors
import gensim.downloader as api

#info = api.info()  # show info about available models/datasets
key2vec = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True) 
weights = key2vec.vectors 

    
    
class Gen_data():
    def __init__(self, data): 
        self.vocab = self.build_vocab(open(data,'r'))
        self.id_vocab = {i:w for i,w in enumerate(self.vocab)}
        self.color_id, self.id_color = self.build_color_ids(open(data,'r'))
        self.shape_id, self.id_shape = self.build_shape_ids(open(data,'r'))

    def build_vocab(self, data):
        vocab = set()
        for line in data:
            line = eval(line)
            vocab.add(line["utter"])
        return list(vocab)

    def build_color_ids(self, envs_data):
        colors = {}

        for inst in envs_data:
            inst = eval(inst)
            for obj in inst["env"]:
                if obj["color"] not in colors.keys():
                    colors.update({obj["color"] : len(colors)})
        id_colors = {i:c for c,i in colors.items()}

        return colors, id_colors


    def build_shape_ids(self, envs_data):
        shapes = {} 

        for inst in envs_data:
            inst = eval(inst)
            for obj in inst["env"]:
                if obj["shape"] not in shapes:
                    shapes.update({obj["shape"] :len(shapes)})
        id_shapes = {i:s for s,i in shapes.items()}
        return shapes, id_shapes


    def build_one_hot_utter(self, utter):  # todo put in a word embedding here        
        return weights[key2vec.vocab[utter].index]        


    def build_one_hot_color(self, color):
        return weights[key2vec.vocab[color].index]        


    def build_one_hot_shape(self, shape):
        return weights[key2vec.vocab[shape].index]        


    def permutate(self, l): #you might have to make this a generator
        return [i for i in multiset_permutations(l)]

        pass #take in necessary objs and their corresponding prob via zip. #create permutations (this could possibly be the first step). use build ftns above to create one_hot_vecs

    def insert_vec_rep(self, env_data):
        target = env_data["target"]
        env_objs = env_data["env"]
        utter = env_data["utter"]

        zipped_objs = [(obj,prob) for obj,prob in zip(env_objs, target)]
        permed_objs = self.permutate(zipped_objs)
        for inst in permed_objs:
            objs, vec, y = self.gen_vec(utter, inst)
            env_data["env"] = objs
            env_data["x"] = [vec]
            env_data["y"] = [y] # why are these in lists? is this hard-coding batch size of 1?
            yield env_data

    def gen_env_vec(self, zipped_objs):
        objects = []
        label = []
        vector = np.array([])
        
        for obj, prob in zipped_objs:
            objects.append(obj)
            label.append(prob)
            
            one_hot_color = self.build_one_hot_color(obj["color"])
            one_hot_shape = self.build_one_hot_shape(obj["shape"])
            vector = np.concatenate((vector, one_hot_color, one_hot_shape), axis=0)
            
        return objects, vector, label
           
    def gen_vec(self, utter, zipped_objs): 
        utter_vec = self.build_one_hot_utter(utter)
        objects, env_vec, label = self.gen_env_vec(zipped_objs)
        return objects, np.concatenate((utter_vec, env_vec), axis=0).tolist(), label
        
if __name__ == "__main__":
    import glob
    dirs = glob.glob('*_objs/')
    
    for d in dirs:
        gener = Gen_data(d + "data.json")
        print(gener.vocab)
        print(gener.color_id)
        print(gener.shape_id)
        data = open(d + "data.json")
        with open(d + "unseen_words_data.json","a") as out:
            for line in data:
                line = eval(line)
                for struct in gener.insert_vec_rep(line):
                    json.dump(struct, out)
                    out.write('\n')
        data.close()
        #os.remove("data.json")
        


#where you left off. Now you're going to store the one hot envs inside the data struct. Most likely you will put in the whole permutation in there.

