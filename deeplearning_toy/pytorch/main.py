import  torch
from    torch import nn, optim
from    torchvision import transforms
from    torchvision import datasets
from    torch.utils.data import DataLoader
# from    LeNet5 import LeNet5
from    ResNet import ResNet


def main():
    batchsz = 64

    cifar_init_train = datasets.CIFAR10('cifar', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_init_train, batch_size=batchsz, shuffle=True)

    cifar_init_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_init_test, batch_size=batchsz, shuffle=True)


    # model = LeNet5()
    model = ResNet()
    print(model)
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    epoch_num = 100
    for epoch in range(epoch_num):
        print('--------- epoch: ', epoch, ' start-------')
        
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            logits = model(x)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if batchidx % 10 == 0:
                print('batchidx: ', batchidx, ' loss: ', loss.item())
        
        print('--------- epoch: ', epoch, ' end-------')
        
        if epoch % 2 == 0:
            torch.save(model.state_dict(), '/home/shiwuwen/workplace/Study/deeplearning_toy/pytorch/model/ResNet-'+str(epoch)+'.pt')
        
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0

            for x, label in cifar_test:
                logits = model(x)
                pred = logits.argmax(dim=1)

                total_correct += torch.eq(pred, label).float().sum().item()

                total_num += x.size(0)
            
            accuracy = total_correct / total_num
            temp = 'epoch: ' + str(epoch) + ' accuracy: ' + str(accuracy)

            print(temp)
            with open('accuracy_txt.txt', 'a') as f:
                f.write(temp)
                f.write('\n')


def test():
    batchsz = 64
    cifar_init_test = datasets.CIFAR10('cifar', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_test = DataLoader(cifar_init_test, batch_size=batchsz, shuffle=True)


    model = ResNet()
    model.load_state_dict(torch.load('/home/shiwuwen/workplace/Study/deeplearning_toy/pytorch/model/ResNet-94.pt'))

    test_correct = 0
    test_total = 0

    for x, y in cifar_test:
        logits = model(x)
        pred = logits.argmax(dim=1)
        test_correct += torch.eq(pred, y).float().sum().item()
        test_total += x.size(0)

    accuracy = test_correct / test_total
    print('test accuracy: ', accuracy)




if __name__ == '__main__':
    # main()
    test()