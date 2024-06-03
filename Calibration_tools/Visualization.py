import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_prices(K, marketPrice, calib_price, name_model):
    plt.subplots(figsize=(10, 5), dpi=100)

    plt.plot(K, marketPrice, label='Market prices', color='black')
    plt.plot(K, calib_price, color='tab:red', label=f'{name_model} model')
            
    plt.xlabel('Strike', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend()
    plt.show()
    
    
def plotCollection(ax, ys, *args, **kwargs):
  ax.plot(ys, *args, **kwargs)

  if "label" in kwargs.keys():

    #remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)

    plt.legend(newHandles, newLabels)