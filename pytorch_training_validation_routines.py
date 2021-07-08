train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
def train_validation(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    valid_loss_min = np.Inf
    
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        train_correct = 0
        valid_correct = 0

        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # sum all the correct predictions to find the training accuracy
            _, predicted_labels = output.max(1)
            train_correct += (predicted_labels == target).float().sum().cpu().numpy()
            # update running training loss
            train_loss += ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            # print updates every 5 batches
            if (batch_idx + 1) % 5 == 0:
                print(f"Epoch: {epoch} \tBatch Index: {batch_idx+1} \tTraining Loss: {train_loss}")

        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # sum all the correct predictions to find the validation accuracy
            _, predicted_labels = output.max(1)
            valid_correct = (predicted_labels == target).float().sum().cpu().numpy()
            # update running validation loss 
            valid_loss += (loss.data.item() - valid_loss) / (batch_idx + 1)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f}% \tValidation Loss: {:.6f} \tValidation Accuracy {:.6f}%'.format(
            epoch,
            train_loss/len(loaders["train"]),
            train_correct/len(loaders["train"])*100,
            valid_loss/len(loaders["valid"]),
            valid_correct/len(loaders["valid"])*100
            ))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    return model
