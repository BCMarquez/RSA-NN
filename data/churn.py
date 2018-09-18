import itertools
import os
import json
import time

colors = ["blue", "green", "purple"]
shapes = ["circle", "square"]


def create_all_objs(colors,shapes):
    all_objs = []
    for c in colors:
        for s in shapes:
            all_objs.append( (c,s) )
    return all_objs


def create_envs(all_objs):
    combs = itertools.combinations(all_objs,3)
    envs = [i for i in combs]
    return envs


env_files_count = 1


def wppl_objs(env):
    repl_tuples = []
    for i in range(len(env)):
        color, shape = env[i][0], env[i][1]
        repl_tuples.append( ( "<obj%d>" % (i+1), "{shape: '%s', color:'%s'}," % (shape,color) ))
    return repl_tuples
    
def wppl_colors(colors):
    repl_tuples = []
    for i in range(len(colors)):
        repl_tuples.append( ("<col%d>" % (i+1), colors[i]))
    return repl_tuples

def color_meaning(colors):
    sub_strs = ["("]
    for i in range(len(colors)):
        sub_strs.append("utterance === '<col%d>'" % (i+1))
        if i != len(colors)-1:
            sub_strs.append(" || ")
    sub_strs.append(")")
    return [("<col_meaning>",''.join(sub_strs))]

def wppl_shapes(shapes):
    repl_tuples = []
    for i in range(len(shapes)):
        repl_tuples.append( ("<sha%d>" % (i+1), shapes[i]))
    return repl_tuples

def shape_meaning(shapes):
    sub_strs = ["("]
    for i in range(len(shapes)):
        sub_strs.append( "utterance === '<sha%d>'" % (i+1))
        if i != len(shapes)-1:
            sub_strs.append(" || ")
    sub_strs.append(")")
    return [("<sha_meaning>",''.join(sub_strs))]

def utter_dist_vars(utter):
    return [("<utter_dist_vars>", "var %s = pragmaticListener('%s')\n" % (utter,utter))]


def json_label_line(utter):
    return [("<label_line>", "json.write('utter_result.json',%s)" % utter)]
    

def create_rsa(env, colors, shapes, utter):
    from shutil import copyfile
    import fileinput
    from functools import reduce
    global env_files_count

    new_file = "rsa_env%d.wppl" % env_files_count
    copyfile("templateRSA.wppl", new_file)
       
    col_meaning = color_meaning(colors)
    sha_meaning = shape_meaning(shapes)

    sliced_utters = [("<utters>", str(colors+shapes))]
    dist_vars = utter_dist_vars(utter)
    write_labels = json_label_line(utter)
    objs = wppl_objs(env)
    c = wppl_colors(colors)
    s = wppl_shapes(shapes)

    repls = col_meaning+sha_meaning+sliced_utters+write_labels+dist_vars+objs+c+s
        
    with open(new_file, 'r+') as myfile:
        s = myfile.read()
        s = reduce(lambda a, kv: a.replace(*kv), repls, s)
        myfile.seek(0)
        myfile.truncate()
        myfile.write(s)
    
    return new_file


def generate_results(rsa):
    command_line_prompt = "webppl %s --require webppl-json webppl-fs" % rsa 
    os.system(command_line_prompt)


#consider passing the utterance in so you can have it in the output file.
def write_results(rsa, zero_dist = None):
    output_path = rsa.replace("wppl","json")
    with open(output_path, "a") as output:
        print("\nabout to write to ", output_path)
        if zero_dist != None:
            print("zero distribution")
            output.write(zero_dist)
        else:
            statinfo = os.stat("utter_result.json")
            file_size = statinfo.st_size
            print("utter_result size: ", file_size) 
            out = open("utter_result.json").read()
            output.write(out+'\n')


def delete_rsa(rsa):
    os.remove(rsa)
    os.remove("utter_result.json")


def create_zero_dist(env):
    sub_strs = ["{'probs':[0,0,0], 'support':["]
    for i in range(len(env)):
        sub_strs.append("{'shape':'%s', 'color':'%s'}" % (env[i][1], env[i][0]))
        if i != len(env)-1: #add comma
            sub_strs.append(",")
    sub_strs.append(']}\n')
    concat_str = ''.join(sub_strs)
    return concat_str.replace("\'","\"")
    
#------------------ process_objs sub-ftns ---------------------------

def find_qualities(env):
    colors = set()
    shapes = set()
    for obj in env:
        colors.add(obj[0])
        shapes.add(obj[1])
    return colors, shapes



def process_objs(environments, utterances):
    global env_files_count
    
    for env in environments:
        colors, shapes = find_qualities(env)

        for u in utterances:
            if u in colors or u in shapes:
                rsa = create_rsa(env, list(colors), list(shapes), u)
                generate_results(rsa)
                write_results(rsa)
                delete_rsa(rsa)

            else: #if utterance describes none of the objects, return a uniform distribution
                rsa = "rsa_env%d.wppl" % env_files_count
                zero_dist = create_zero_dist(env) 
                write_results(rsa, zero_dist)

        env_files_count += 1


def create_utterance_file(colors, shapes):
    with open("pragmaticTrainUtter.json", "w") as train_utter:
        string = "{'colors': %s, 'shapes':%s}" % (str(colors),str(shapes))
        train_utter.write(string)
    


if __name__ == "__main__":
    create_utterance_file(colors,shapes)
    utterances = colors+shapes
    all_objs = create_all_objs(colors, shapes)
    envs = create_envs(all_objs)
    process_objs(envs,utterances)

