import pickle
from pathlib import Path
import matplotlib.pyplot as plt


losses_save_path = 'models'
losses_file_name_FB = "MDRNone-freebits0.005-word_dropout0.0-print_every50-iterations9952"
save_losses_path_FB = Path(losses_save_path) / losses_file_name_FB
with open(save_losses_path_FB, 'rb') as file:
    (lists_FB, print_every_FB, args) = pickle.load(file)

#Add more lists here and below by repeating the plot()s once we have all the results

fig, ax1 = plt.subplots()
x = [print_every_FB * x_ for x_ in range(len(lists_FB[0]))]
ax1.plot(x, lists_FB[0], label = 'NLL')

ax2 = ax1.twinx()
ax2.plot(x, lists_FB[1], label = 'KL')

savefile = 'test_plot.png'

open(savefile, 'w').close()
plt.savefig(savefile, bbox_inches='tight')