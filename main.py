from functions import *
from mnist import *
from network import *

if __name__ == "__main__":
    x_train, t_train, x_test, t_test = get_data(one_hot_label=False)

    train_size = x_train.shape[0]
    batch_size = 100

    print(t_train[1000])

    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # net = read_network('sample_weight.pkl')
    # accuracy_cnt = 0
    # for i in range(len(x)):
    #     y = predict(net, x[i])
    #     p = np.argmax(y)  # takes out the index 0 to 9 here.
    #     if p == t[i]:
    #         accuracy_cnt += 1
    # print('Accuracy : {}'.format(float(accuracy_cnt) / len(t)))
