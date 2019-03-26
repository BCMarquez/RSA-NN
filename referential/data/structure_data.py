#this file only structure the data into one json structure

def doc_to_list(path):
    """Function reads in json data from rsa, converts it to a dict, and stores it in a list"""
    txtFile = open(path,"r")
    lines = list()
    for line in txtFile:
        lines.append(eval(line))
    txtFile.close()
    return lines


def objs_in_env(env_label_data):
    world_objs = []
    for label in env_label_data:
        for obj in label["support"]:
            if obj not in world_objs:
                world_objs.append(obj)
    return world_objs


def add_string_rep(world_objs):
    for obj in world_objs:
        obj.update({"string": obj["color"] + " " + obj["shape"]}) #add string representation of object
    return world_objs


def apply_utter(identif, utterance, environment, env_label_data):
    data = {"id":identif,
            "utter": "",
            "env": [],
            "target": []
            }
    data["utter"] = utterance
    data["env"] = environment
    probs = [0] * len(environment)
    for prob, obj in zip(label["probs"], label["support"]):
        index = environment.index(obj)
        probs[index] = prob
    data["target"] = probs
    return data


if __name__ == "__main__":
    #code that goes through the files generated by churn
    import json
    import os
    import glob

    dirs = glob.glob('rsa_env*')

    with open('data.json','a') as out:
        for identif,env_path in enumerate(dirs):
            env_label_data = doc_to_list(env_path)
            env_objs = objs_in_env(env_label_data)
            for utter, label in zip(utterances, env_label_data):
                data = apply_utter(identif, utter, env_objs, label)
                if sum(data["target"]) != 0: #filter out instances where utterance doesn't correspond to objects
                    json.dump(data,out)
                    out.write('\n')
            os.remove(env_path)
