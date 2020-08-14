import pickle
import keras
import matplotlib.pyplot as plt



def DocReader(path):

    file = open(path,'rb')
    data_train =  pickle.load(file)
    file.close()
    history_dict = data_train
    print(history_dict.keys())

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs_as_list = range(1, len(loss_values) + 1)

    print(type(epochs_as_list))

    plt.style.use('seaborn-darkgrid')

    train_loss_line = plt.plot(epochs_as_list, loss_values, label='Train loss')
    test_loss_line = plt.plot(epochs_as_list, val_loss_values, label='Validation/Test loss')

    plt.setp(train_loss_line, linewidth=2.0, marker='*', markersize=5.0)
    plt.setp(test_loss_line, linewidth=2.0, marker='*', markersize=5.0)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    train_acc_line = plt.plot(epochs_as_list, acc_values, label='Train accuracy')
    test_acc_line = plt.plot(epochs_as_list, val_acc_values, label='Test accuracy')

    plt.setp(train_acc_line, linewidth=2.0, marker='*', markersize=5.0)
    plt.setp(test_acc_line, linewidth=2.0, marker='*', markersize=5.0)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

    #y_pred = model.predict_classes(I_test)

    #print(classification_report(np.argmax(L_test, axis=1), y_pred))

    #cm = confusion_matrix(np.argmax(L_test, axis=1), y_pred)  # np.argmax because our labels were one hot encoded
    #plt.figure(figsize=(20, 10))
    #sns.heatmap(cm, annot=True)

def main():
    DocReader('D:/Task01_BrainTumour/Mono_27_basic_Ephos50_Slice2_extended/trainHistoryDict')

if __name__ == "__main__":
    # execute only if run as a script
    main()