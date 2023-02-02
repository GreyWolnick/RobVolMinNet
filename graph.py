import matplotlib.pyplot as plt

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='none', sharey='none')

# ax1.plot()
ax1.set_ylabel('Testing Accuracy')
ax1.set_xlabel('Epoch')
# ax1.set_title('MNIST | Noise = 0.20')
ax1.set_ylim([0, 1])
# ax1.legend(bbox_to_anchor=(2.45, 1.5))

# ax2.plot()
ax2.set_ylabel('Training Loss')
ax2.set_xlabel('Epoch')
# ax2.set_title('CIFAR10 | Noise = 0.20')
ax2.set_ylim([0, 1])

# ax3.plot()
ax3.set_ylabel('T Estimation Error')
ax3.set_xlabel('Epoch')
# ax3.set_title('CIFAR10 | Noise = 0.20')
ax3.set_ylim([0, 1])

plt.tight_layout()
plt.show()












