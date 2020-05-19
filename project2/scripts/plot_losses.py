import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import numpy as np

losses_save_path = 'losses'

## FreeBits
# losses_file_name = "MDRNone-freebits0.005-word_dropout0.0-print_every50-iterations9952"
# save_losses_path = Path(losses_save_path) / losses_file_name_FB
# with open(save_losses_path, 'rb') as file:
#     (lists_FB, print_every, args) = pickle.load(file)

## Vanilla
losses_file_name = "VANILLA-MDRNone-freebitsNone-word_dropout0.0-print_every50-iterations4976"
save_losses_path = Path(losses_save_path) / losses_file_name
with open(save_losses_path, 'rb') as file:
    (lists_vanilla, print_every, args) = pickle.load(file)
print(print_every)

## Word dropout
losses_file_name = "DROPOUT-MDRNone-freebitsNone-word_dropout0.66-print_every50-iterations4976"
save_losses_path = Path(losses_save_path) / losses_file_name
with open(save_losses_path, 'rb') as file:
    (lists_dropout, print_every, args) = pickle.load(file)
print(print_every)

## MDR
losses_file_name = "MDR - MDR5.0-freebitsNone-word_dropout0.66-print_every50-iterations4976"
save_losses_path = Path(losses_save_path) / losses_file_name
with open(save_losses_path, 'rb') as file:
    (lists_mdr, print_every, args) = pickle.load(file)
print(print_every)

#Add more lists here and below by repeating the plot()s once we have all the results

x = [print_every * x_ for x_ in range(len(lists_vanilla[0]))]
x_smooth = np.linspace(min(x), max(x), 100)
# y_smooth = spline(x, y, x_smooth)

# a = 
nll_vanilla_smooth = interpolate.make_interp_spline(x, lists_vanilla[0])(x_smooth)
nll_dropout_smooth = interpolate.make_interp_spline(x, lists_dropout[0])(x_smooth)
nll_mdr_smooth = interpolate.make_interp_spline(x, lists_mdr[0])(x_smooth)

# f = interpolate.interp1d(x, lists_dropout[0], 'cubic')

fig, ax1 = plt.subplots()
ax1.plot(x_smooth, nll_vanilla_smooth, '--', label = 'Vanilla - NLL', color='blue')
ax1.plot(x_smooth, nll_dropout_smooth, '--', label = 'Dropout - NLL', color='red')
ax1.plot(x_smooth, nll_mdr_smooth, '--', label = 'MDR - NLL', color='orange')


ax1.set_ylim(0,200)
ax1.set_ylabel("Training NLL")

kl_vanilla_smooth = interpolate.make_interp_spline(x, lists_vanilla[1])(x_smooth)
kl_dropout_smooth = interpolate.make_interp_spline(x, lists_dropout[1])(x_smooth)
kl_mdr_smooth = interpolate.make_interp_spline(x, lists_mdr[1])(x_smooth)


ax2 = ax1.twinx()
ax2.plot(x_smooth, kl_vanilla_smooth, label = 'Vanilla - KL', color='blue')
ax2.plot(x_smooth, kl_dropout_smooth, label = 'Dropout - KL', color='red')
ax2.plot(x_smooth, kl_mdr_smooth, label = 'MDR - KL', color='orange')


ax2.set_ylabel('Training KL Divergence')
ax2.set_xlabel("Training iterations")
ax2.set_ylim(0,15)

fig.tight_layout()
fig.legend(fancybox = True, framealpha = 0.9, loc =  'upper right')

savefile = 'test_plot.png'

open(savefile, 'w').close()
plt.savefig(savefile, bbox_inches='tight')