import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from net import my_net
from loss import MyLoss
from data import MyDataset
from torch.autograd import Variable

'''
Log in tensorboard
'''
curve_train_loss = []
curve_test_loss = []
curve_precision = []
curve_speed_per_image = []
tb = SummaryWriter('../train/log')

'''
Configuration
'''

learning_rate = 1e-3
num_epochs = 50
BATCH_SIZE = 4
train_dir = '../dataset/train'
test_dir = '../dataset/test'
print('Training params: learning_rate = {0} , num_epochs = {1} , BATCH_SIZE = {2}.'.format(learning_rate, num_epochs,
                                                                                           BATCH_SIZE))

'''
Show net
'''

print(my_net)

'''
Loss Function
'''

criterion = MyLoss(7, 2, 5.0, 0.5)
optimizer = torch.optim.SGD(
    my_net.parameters(), learning_rate, momentum=0.9, weight_decay=5e-4
)

'''
Train
'''

train_set = MyDataset(train_dir, 'img', 'label')
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=2)
test_set = MyDataset(test_dir, 'img', 'label')
test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False, num_workers=2)
print('Train dataset contains %d images' % (len(train_set)))
print('Test dataset contains %d images' % (len(test_set)))

'''
Start Training
'''
best_test_loss = 10.0
for epoch in range(num_epochs):
    my_net.train()
    # change learning rate
    if epoch == 30:
        learning_rate = 1e-4
    if epoch == 40:
        learning_rate = 1e-5
    # update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    train_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, targets = data
        inputs = Variable(inputs)
        targets = Variable(targets)

        optimizer.zero_grad()
        outputs = my_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]

        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, average_loss: %.4f' % (epoch + 1, num_epochs,
                                                                                   i + 1, len(train_loader),
                                                                                   loss.data[0], train_loss / (i + 1)))
    tb.add_scalar(tag='train loss', scalar_value=train_loss / (len(train_loader)), global_step=epoch)
    # curve_train_loss.append(train_loss / (len(train_loader)))

    # validation per epoch

    validation_loss = 0.0
    my_net.eval()
    for i, data in enumerate(test_loader):
        inputs, targets = data
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)

        outputs = my_net(inputs)
        loss = criterion(outputs, targets)
        validation_loss += loss.data[0]
    validation_loss = validation_loss / len(test_loader)
    # curve_test_loss.append(validation_loss)
    if best_test_loss < validation_loss:
        best_test_loss = validation_loss
        print('Get best loss at epoch:{}, the best loss is {}.'.format(epoch, best_test_loss))
        torch.save(my_net.state_dict(), '../train/module/best.pth')
    tb.add_scalar(tag='validation loss', scalar_value=validation_loss, global_step=epoch)
    torch.save(my_net.state_dict(), '../train/module/last.pth')
tb.close()
