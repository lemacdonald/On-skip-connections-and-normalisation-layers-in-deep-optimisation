import numpy as np
import matplotlib.pyplot as plt

vals = [0.2, 0.1, 0.05, 0.02]

for v in vals:
    x = np.load(f'comparison_100_loss_90epochs_datanormalised_{v:g}.npy')
    # x = np.load(f'comparison_100_test_90epochs_datanormalised_{v:g}.npy')
    # if v == 0.1:
    #     x = np.load('comparison_10_loss_90epochs_datanormalised.npy')[:,:10]
    # else:
    #     x = np.load(f'comparison_10_loss_90epochs_datanormalised_{v:g}.npy')

    # if v == 0.2:
    #     x = np.delete(x, [0, 2, 4, 5, 6, 7], 1)

    # print('original: ', x[0].mean(), x[0].std())
    # print('modified: ', x[1].mean(), x[1].std())

    print('original: ', x[0,:,-391:].mean(axis = -1).mean(), x[0,:,-391:].mean(axis = -1).std())
    print('modified: ', x[1,:,-391:].mean(axis = -1).mean(), x[1,:,-391:].mean(axis = -1).std())

    # y = np.zeros((2, 10, 90))
    # # print(x.shape)

    # for i in range(90):
    #     y[:,:,i] = x[:,:,i*391:(i+1)*391].mean(-1)

    # if v == 0.2:
    #     y = np.delete(y, [0, 2, 4, 5, 6, 7], 1)
    # # if v == 0.1:
    # #     y = np.delete(y, 4, 1)
    # print(0, y[0,:,-1].mean())
    # print(1, y[1,:,-1].mean())

    # ym = y.mean(axis = 1)
    # ystd = y.std(axis = 1)

    # t = np.linspace(0, 90, 90)
    # xticks = np.array([0, 20, 40, 60, 80])

    # fig, ax = plt.subplots()
    # ax.plot(t, ym[0], label = 'orig.', linestyle = 'solid', linewidth = 2.0)
    # ax.fill_between(t, ym[0]-ystd[0], ym[0]+ystd[0], alpha = 0.3)
    # ax.plot(t, ym[1], label = 'mod.', linestyle = 'dashed', linewidth = 2.0)
    # ax.fill_between(t, ym[1]-ystd[1], ym[1]+ystd[1], alpha = 0.3)
    # ax.legend(fontsize = 26)
    # ax.set_yscale('log')
    # plt.xticks(xticks, fontsize = 26)
    # plt.xlabel('Epoch', fontsize = 26)
    # plt.ylabel('Loss', fontsize = 26)
    # plt.yticks(fontsize = 26)
    # fig.tight_layout()
    # plt.savefig(f'./cifar10_{v:g}.pdf')
    # plt.show()

# s = np.load('svals2conv_trials_nonzinit.npy')
# print(s.shape) #(10, 3, 50, 320)
# s = s[:,:,:10,:]

# s = s.mean(axis=0)
# print(s.shape)
# s = s.mean(axis=1)
# print(s.shape)

# xticks = np.array([0.0, 2.0, 4.0, 6.0])
# yticks = np.array([0, 10, 20, 30, 40])


# for i in range(3):
#     plt.hist(s[i], 32, (0.0, 6.0))
#     plt.xticks(xticks, fontsize = 30)
#     plt.xlabel('Sing. Val.', fontsize = 30)
#     plt.ylabel('Frequency', fontsize = 30)
#     plt.ylim((0, 40))
#     plt.yticks(yticks, fontsize = 30)
#     plt.tight_layout()
#     plt.savefig('sing{}.pdf'.format(i))
#     plt.show()

# l = np.load('lossvalsconv_trials_nonzinit.npy')
# print(l.shape)
# l = l[:,:,:20]
# l = l.mean(axis = 0)

# xticks = np.array([0, 10, 20])
# yticks = np.array([0.0, 1.25, 2.5])

# t = np.linspace(0, 20, 20)
# plt.plot(t, l[0], label = '(a)', linestyle = 'solid', linewidth = 3.0)
# plt.plot(t, l[1], label = '(b)', linestyle = 'dashed', linewidth = 3.0)
# plt.plot(t, l[2], label = '(c)', linestyle = 'dotted', linewidth = 3.0)
# plt.xlabel('Epoch', fontsize = 30)
# plt.ylabel('Loss', fontsize = 30)
# plt.xticks(xticks, fontsize = 30)
# plt.yticks(yticks, fontsize = 30)
# plt.legend(fontsize = 30)
# plt.tight_layout()
# plt.savefig('loss.pdf')
# plt.show()

# A = 2*(np.random.rand(500, 500) - 0.5)/np.sqrt(500)
# IpA = np.eye(500) + A

# s = np.zeros((2, 500))
# s[0] = np.linalg.svd(A)[1]
# s[1] = np.linalg.svd(IpA)[1]

# xticks = np.array([0.0, 1.0, 2.0])
# yticks = np.array([0, 20, 40])

# for i in range(2):
#     plt.hist(s[i], 30, (0.0, 2.0))
#     plt.xticks(xticks, fontsize = 30)
#     plt.xlabel('Sing. Val.', fontsize = 30)
#     plt.ylabel('Frequency', fontsize = 30)
#     plt.ylim((0, 40))
#     plt.yticks(yticks, fontsize = 30)
#     plt.tight_layout()
#     plt.savefig('randsing{}.pdf'.format(i))
#     plt.show()
