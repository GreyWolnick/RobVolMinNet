import matplotlib.pyplot as plt

mnist_gce_005_test = [0.9461, 0.9673, 0.9803, 0.9825, 0.9798, 0.9838, 0.9848, 0.9854, 0.9857, 0.9868, 0.986, 0.9881, 0.9857, 0.9844, 0.9855, 0.9859, 0.9849, 0.9857, 0.9847, 0.9848, 0.9847, 0.984, 0.9837, 0.9849, 0.9836, 0.9839, 0.9851, 0.9835, 0.9851, 0.9835, 0.9834, 0.9829, 0.9825, 0.9818, 0.9817, 0.9837, 0.9831, 0.9828, 0.9812, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914, 0.8914]

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='none', sharey='none')

ax1.plot(mnist_gce_005_test, label="GCE")
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












