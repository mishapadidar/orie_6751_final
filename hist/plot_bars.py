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
filelist.sort()

# only samples under these perturbations will be plotted
plot_nps   = 0.002
plot_shift = 0.002

# convert from meters to mm
to_mm = lambda x: int(1000*x)

# for labeling
problem_names = ['CVaR','Expected']

# generate a dataframe
data = []
for infile in filelist:
  # attributes
  attrib = {}
  # load file
  indata = pickle.load(open(infile,"rb"))

  # filter so we only plot the right sample sizes
  if indata['sample_max_shift'] != plot_shift or indata['sample_normal_perturbation_size'] != plot_nps:
    continue

  # set the label
  if 'max_shift' in indata.keys():
    label = problem_names[indata['problem_num']]\
    +"-"+ str(to_mm(indata['max_shift']))\
    +"-"+ str(to_mm(indata['normal_perturbation_size']))
  else:
    label = str(indata['problem_num']) 

  attrib['run'] = label
  attrib['mean']  = indata['mean']
  attrib['std']   = indata['std']
  attrib['var99']  = indata['var99']
  attrib['var95']  = indata['var95']
  attrib['cvar95'] = indata['cvar95']
  attrib['cvar99'] = indata['cvar99']
  attrib['max']  = indata['max']
  attrib['min']  = indata['min']
  attrib['bnorm']  = indata['bnorm']
  data.append(attrib)

df = pd.DataFrame.from_dict(data)
df = df.sort_values(by=['var95'])
print(df)

# Set up the matplotlib figure
# f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 5), sharex=True)
f, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, figsize=(7, 5), sharex=True)

sns.barplot(x="run", y="var95", palette="rocket", data=df,ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("95%-Quantile")
sns.barplot(x="run", y="bnorm", palette="rocket", data=df,ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("$f_B$")
sns.barplot(x="run", y="mean", palette="rocket", data=df,ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("Mean")
sns.barplot(x="run", y="cvar95", palette="rocket", data=df,ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
ax4.set_ylabel("CVaR95")

ax1.title.set_text(f"Objectives under {to_mm(plot_shift)}mm alignment and {to_mm(plot_nps)}mm fabrication errors")
plt.show()

