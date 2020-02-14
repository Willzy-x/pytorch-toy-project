from torch.autograd import Variable
import torch
# COUNT_BATCH = 40


def train(epoch, net, optimizer, criterion, train_loader, train_file):
    total_images = 0
    correct_predictions = 0
    running_loss = 0
    data_size = len(train_loader)
    for i, data in enumerate(train_loader, 0):
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = net(images)
        _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_images += labels.size(0)
        correct_predictions += (predictions == labels).sum().item()

        loss_data = loss.item()
        running_loss += loss_data

    train_accuracy = 100 * correct_predictions / total_images
    avg_loss = running_loss / data_size
    print("Training: Epoch: [%d], trained: [%5d], Loss: [%.6f], Training Accuracy: [%.5f]" % (
        epoch, total_images, avg_loss, train_accuracy
    ))
    train_file.write("{},{},{}\n".format(epoch, avg_loss, train_accuracy))

    return avg_loss, train_accuracy


def test(epoch, net, criterion, test_loader, test_file):
    total_images = 0
    correct_predictions = 0
    running_loss = 0
    data_size = len(test_loader)
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            outputs = net(images)
            _, predictions = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            total_images += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()

            loss_data = loss.item()
            running_loss += loss_data

            test_accuracy = 100 * correct_predictions / total_images
            avg_loss = running_loss / data_size

    print("Test: Epoch: [%d], tested: [%5d], Loss: [%.6f], Test Accuracy: [%.5f]" % (
        epoch, total_images, avg_loss, test_accuracy
    ))
    test_file.write("{},{},{}\n".format(epoch, avg_loss, test_accuracy))

    return avg_loss, test_accuracy
