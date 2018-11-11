import json
import matplotlib.pyplot as plt


marker = [',', '+', '.', 'o', '*']
nb_epoch = 100


def plot_train_acc(i, historyList, nameList, title):
    fig = plt.figure()
    for index, his in enumerate(historyList):
        plt.plot(range(nb_epoch),his['acc'],label=nameList[index],marker = marker[index])
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('training_accuracy')
    plt.xlim([1,nb_epoch])
    plt.grid(True)
    plt.title("Training Accuracy Comparison "+title)
    #plt.show()
    fig.savefig('img/'+str(i)+'-training-accuracy.png')
    plt.close(fig)
    
def plot_val_acc(i, historyList, nameList, title):
    fig = plt.figure()
    for index, his in enumerate(historyList):
        plt.plot(range(nb_epoch),his['val_acc'],label=nameList[index],marker = marker[index])
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('validation_accuracy')
    plt.xlim([1,nb_epoch])
    plt.grid(True)
    plt.title("Validation Accuracy Comparison "+title)
    #plt.show()
    fig.savefig('img/'+str(i)+'-validation-accuracy.png')
    plt.close(fig)

 
historyDef = json.load(open('historyDef.json'))
history1 = json.load(open('history1.json'))
history2 = json.load(open('history2.json'))
history3 = json.load(open('history3.json'))
history4 = json.load(open('history4.json'))
history5 = json.load(open('history5.json'))
history6 = json.load(open('history6.json'))
history7 = json.load(open('history7.json'))
history8 = json.load(open('history8.json'))
history9 = json.load(open('history9.json'))
historyFINAL = json.load(open('historyFINAL.json'))

historyDef_CNN = json.load(open('historyDef_CNN.json'))
history1_CNN = json.load(open('history1_CNN.json'))
history3_CNN = json.load(open('history3_CNN.json'))
history4_CNN = json.load(open('history4_CNN.json'))
history6_CNN = json.load(open('history6_CNN.json'))

plot_train_acc(1, [historyDef, history1, history2], ['default MLP (512, 256)', '(256, 128) MLP', '(1024, 512) MLP'], '(number of neurons)')
plot_val_acc(2, [historyDef, history1, history2], ['default MLP (512, 256)', '(256, 128) MLP', '(1024, 512) MLP'], '(number of neurons)')

plot_train_acc(3, [historyDef, history3, history4], ['default MLP (0.2-drop_rate)', '0.1-drop_rate MLP', '0.5-drop_rate MLP'], '(drop rate)')
plot_val_acc(4, [historyDef, history3, history4], ['default MLP (0.2-drop_rate)', '0.1-drop_rate MLP', '0.5-drop_rate MLP'], '(drop rate)')

plot_train_acc(5, [historyDef, history5],['default MLP (2 hidden layers)', '3 hidden layers MLP'], '(number of layers)')
plot_val_acc(6, [historyDef, history5],['default MLP (2 hidden layers)', '3 hidden layers MLP'], '(number of layers)')

plot_train_acc(19, [historyDef, historyFINAL],['default MLP ', 'my MLP'], '')
plot_val_acc(20, [historyDef, historyFINAL],['default MLP', 'my MLP'], '')

plot_train_acc(17, [historyDef, history6, history7], ['default MLP (lr=0.01,dc=0.0)', '(lr=0.03,dc=0.0) MLP', '(lr=0.01,dc=0.0001) MLP'], '(learning rate)')
plot_val_acc(18, [historyDef, history6, history7], ['default MLP (lr=0.01,dc=0.0)', '(lr=0.03,dc=0.0) MLP', '(lr=0.01,dc=0.0001) MLP'], '(learning rate)')

plot_train_acc(9, [historyDef, history8, history9], ['default MLP (relu)', 'sigmoid MLP', 'softplus MLP'], '(activation function)')
plot_val_acc(10, [historyDef, history8, history9], ['default MLP (relu)', 'sigmoid MLP', 'softplus MLP'], '(activation function)')

plot_train_acc(11, [historyDef_CNN, history1_CNN], ['default CNN (no normalization)', 'CNN with Batch normalization'], '(batch normalization)')
plot_val_acc(12, [historyDef_CNN, history1_CNN], ['default CNN (no normalization)', 'CNN with Batch normalization'], '(batch normalization)')

plot_train_acc(13, [historyDef_CNN, history3, history4], ['default 0.25, 0.5 drop_rate CNN', '0.1, 0.3 drop_rate CNN', '0.5, 0.5 drop_rate CNN'], '(drop rate)')
plot_val_acc(14, [historyDef_CNN, history3, history4], ['default 0.25, 0.5 drop_rate CNN', '0.1, 0.3 drop_rate CNN', '0.5, 0.5 drop_rate CNN'], '(drop rate)')

plot_train_acc(30, [historyDef_CNN, history6_CNN], ['default (lr=0.01) CNN', '(lr=0.03) CNN'], '(learning rate)')
plot_val_acc(31, [historyDef_CNN, history6_CNN], ['default (lr=0.01) CNN', '(lr=0.03) CNN'], '(learning rate)')

nameList = ['default (lr=0.01) CNN', '(lr=0.03) CNN']

fig = plt.figure()
for index, his in enumerate([historyDef_CNN, history6_CNN,]):
    plt.plot(range(nb_epoch),his['val_loss'],label=nameList[index],marker = marker[index])
plt.legend(loc=0)
plt.xlabel('epochs')
plt.ylabel('validation_loss')
plt.xlim([1,nb_epoch])
plt.grid(True)
plt.title("Validation Loss Comparison "+'(learning rate)')
#plt.show()
fig.savefig('img/60-validation-loss.png')
plt.close(fig)


nameList = ['default (lr=0.01,dc=0.0) MLP', '(lr=0.03,dc=0.0) MLP', '(lr=0.01,dc=0.0001) MLP']

fig = plt.figure()
for index, his in enumerate([historyDef, history6, history7]):
    plt.plot(range(nb_epoch),his['val_loss'],label=nameList[index],marker = marker[index])
plt.legend(loc=0)
plt.xlabel('epochs')
plt.ylabel('validation_loss')
plt.xlim([1,nb_epoch])
plt.grid(True)
plt.title("Validation Loss Comparison "+'(learning rate)')
#plt.show()
fig.savefig('img/50-validation-loss.png')
plt.close(fig)
