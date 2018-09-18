from sklearn.utils import shuffle
import math
import random
import json
import os

train = "train.json"
dev   = "dev.json"
test  = "test.json"

class Splitter:
    def __init__(self, data):
        self.data_path = data

#----------------- hidden env ------------------------------
    def hidden_env(self, envs_to_hide):
        data = open(self.data_path,"r")
        envs_to_hide.sort()
        for line in data:
            line = eval(line)
   
            
            if len(envs_to_hide) > 0 and envs_to_hide[0] < line["id"]:
                envs_to_hide.pop(0)
   
            if len(envs_to_hide) > 0 and envs_to_hide[0] == line["id"]:
                    print("id match")
                    self.write_to_file(test,[line])
            else:
                self.write_to_file("temp.json", [line])
        self.train_dev_split("temp.json")
        os.remove("temp.json")


#----------------- hidden obj ------------------------------
    def objs_as_strs(self, env_data):
        objs = []
        for obj in env_data["env"]:
            objs.append("%s %s"%(obj["color"], obj["shape"]))
        return objs
    

    def hidden_obj(self, obj_to_hide):
        data = open(self.data_path,"r")
        for line in data:
            line = eval(line)
            str_env = self.objs_as_strs(line)
            if obj_to_hide in str_env:
                self.write_to_file(test,[line])
            else:
                self.write_to_file("temp.json", [line])
        self.train_dev_split("temp.json")
        os.remove("temp.json")


#I feel like this function is not necessary.
    def train_dev_split(self, path):
        total_envs = self.doc_to_list(path)
        random.seed(230)
        print("total envs: ", type(total_envs))
        random.shuffle(total_envs)

        split = int(.8 * len(total_envs))
        
        self.write_to_file(train, total_envs[:split])
        self.write_to_file(dev, total_envs[split:])
    
#-------------- random split ------------------------------
    def random_split(self):
        total_envs = self.doc_to_list(self.data_path)
        random.seed(230)
        random.shuffle(total_envs)

        split_1 = int(.8 * len(total_envs))
        split_2 = int(.9 * len(total_envs))
        
        self.write_to_file(train, total_envs[:split_1])
        self.write_to_file(dev, total_envs[split_1:split_2])
        self.write_to_file(test,total_envs[split_2:])

#----------------------- helper functions -------------------------
    def write_to_file(self, path, data):
        with open(path,'a') as out:
            for ele in data:
                json.dump(ele, out)
                out.write('\n')
        

    def doc_to_list(self, path):
        """Function reads in json data from rsa, converts it to a dict, and stores it in a list"""
        txtFile = open(path,"r")
        instances = list()
        for line in txtFile:
            instances.append(eval(line))
        txtFile.close()
        return instances
    

if __name__ == "__main__":
    splitter = Splitter("final_data.json")
    #splitter.random_split()
    splitter.hidden_env([4])
    #splitter.hidden_obj("purple square")

