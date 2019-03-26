import glob
import matplotlib.pyplot as plt
import json


def create_all_objs(colors,shapes):
    all_objs = []
    for c in colors:
        for s in shapes:
            all_objs.append( (c,s) )
    return all_objs

def create_graphs(dirs):
    x_axis = []
    total_time_y_axis = []
    avg_time_y_axis = []

    for d in dirs:
        print(d) 
        with open(d+"pragmaticTrainUtter.json", "r") as json_data:
            data = eval(json_data.read())
        all_objs = create_all_objs(data["colors"], data["shapes"])
        num_of_objs = len(all_objs)
        x_axis.append(num_of_objs)

        with open(d+"timing.json", "r") as timing:
            data = eval(str(json.load(timing)))
            print(data)
        total_time_y_axis.append(data[0]["total_time"])
        avg_time_y_axis.append(data[0]["avg_pred_time"])
        
    return (x_axis, total_time_y_axis), (x_axis, avg_time_y_axis)


dirs = glob.glob('runs/*_objs/')
total_time, avg_time = create_graphs(dirs)


print("total_time: ", total_time)
print("avg_time: ", avg_time)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(total_time[0], total_time[1], color='lightblue', linewidth=3)
ax1.set(title="Total Time", ylabel='seconds', xlabel='# of objs')


ax2 = fig.add_subplot(212)
ax2.plot(avg_time[0], avg_time[1], color='orange', linewidth=3)
ax2.set(title="Avg Time", ylabel='seconds', xlabel='# of objs')

plt.savefig('5_envs_graphs.png')
plt.show()
