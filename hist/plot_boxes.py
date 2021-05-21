import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import pandas as pd
import numpy as np
import glob
# sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 18})


# load the data
filelist = glob.glob("./output/*.pickle")

# only samples under these perturbations will be plotted
plot_nps   = 0.002
plot_shift = 0.002
to_mm = lambda x: int(1000*x)

# generate a dataframe
fX = []
runs = []

problem_names = ['CVaR','Expected']
for infile in filelist:
  # attributes
  attrib = {}
  # load file
  indata = pickle.load(open(infile,"rb"))

  # filter so we only plot the right sample sizes
  if indata['sample_max_shift'] != plot_shift or indata['sample_normal_perturbation_size'] != plot_nps:
    continue

  fX.append(indata['fX_samples'])
  if 'max_shift' in indata.keys():
    label = problem_names[indata['problem_num']]\
    +"-"+ str(to_mm(indata['max_shift']))\
    +"-"+ str(to_mm(indata['normal_perturbation_size']))
  else:
    label = str(indata['problem_num']) 
  runs.append(label)

# sort the labels
fX  = np.array(fX)
idx = np.argsort(runs)
runs.sort()
fX = fX[idx].T

fig = plt.figure(figsize=(12,8))
ax = plt.axes()

# box plot
bp0 = plt.boxplot(fX,patch_artist=True)
colors = sns.color_palette('colorblind')
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp0[element], color='k',linewidth=2)
for ii,box in enumerate(bp0['boxes']):
    box.set(facecolor = colors[ii] )

ylocs,ylabs = plt.yticks()
print(ylocs,ylabs)
plt.yticks(ylocs[::2])
plt.xticks(range(1,len(runs)+1),runs,rotation=0)
plt.ylabel("Field Error")
# plt.legend(loc=1)
plt.title(f"Distribution under {to_mm(plot_shift)}mm alignment and {to_mm(plot_nps)}mm fabrication errors")
plt.show()

