from methods import *
import os, shutil

#user inputs

#load hyperparameters
sizes = ['1_tiny', '2_small', '3_standard', '4_full']
size_folders = ['size_data_f1/' + size for size in sizes]

#datasets
datasets = ['sst2']

#number of output classes
num_classes_list = [2, 2, 2, 6, 2]

#number of augmentations per original sentence
n_aug_list_dict = {'size_data_f1/1_tiny': [32, 32, 32, 32, 32], 
					'size_data_f1/2_small': [32, 32, 32, 32, 32],
					'size_data_f1/3_standard': [16, 16, 16, 16, 4],
					'size_data_f1/4_full': [16, 16, 16, 16, 4]}

if not os.path.isdir('size_data_f1'):
    os.mkdir('size_data_f1')
#number of words for input
input_size_list = [50, 50, 40, 25, 25]

alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# Number of words for input
input_size_list = [50,50,40,25,25]

#word2vec dictionary
huge_word2vec = 'word2vec/glove.840B.300d.txt'
word2vec_len = 300

def run_cnn(train_file, test_file, num_classes, percent_dataset):

    #initialize model
    model = build_cnn(input_size, word2vec_len, num_classes)

    #load data
    train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec, percent_dataset)
    test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec, 1)

    #implement early stopping
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    #train model
    model.fit(	train_x, 
                train_y, 
                epochs=20, 
                callbacks=callbacks,
                validation_split=0.1, 
                batch_size=1024, 
                shuffle=True, 
                verbose=1)
    #model.save('checkpoints/lol')
    #model = load_model('checkpoints/lol')

    #evaluate model
    y_pred = model.predict(test_x)
    test_y_cat = one_hot_to_categorical(test_y)
    y_pred_cat = one_hot_to_categorical(y_pred)
    acc = accuracy_score(test_y_cat, y_pred_cat)

    #clean memory???
    train_x, train_y, test_x, test_y, model = None, None, None, None, None
    gc.collect()

    #return the accuracy
    #print("data with shape:", train_x.shape, train_y.shape, 'train=', train_file, 'test=', test_file, 'with fraction', percent_dataset, 'had acc', acc)
    return acc

def run_rnn(train_file, test_file, num_classes, percent_dataset):

    #initialize model
    model = build_model(input_size, word2vec_len, num_classes)

    #load data
    train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec, percent_dataset)
    test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec, 1)

    #implement early stopping
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

    #train model
    model.fit(	train_x, 
                train_y, 
                epochs=20, 
                callbacks=callbacks,
                validation_split=0.1, 
                batch_size=1024, 
                shuffle=True, 
                verbose=1)
    #model.save('checkpoints/lol')
    #model = load_model('checkpoints/lol')

    #evaluate model
    y_pred = model.predict(test_x)
    test_y_cat = one_hot_to_categorical(test_y)
    y_pred_cat = one_hot_to_categorical(y_pred)
    acc = accuracy_score(test_y_cat, y_pred_cat)

    #clean memory???
    train_x, train_y, test_x, test_y, model = None, None, None, None, None
    gc.collect()

    #return the accuracy
    #print("data with shape:", train_x.shape, train_y.shape, 'train=', train_file, 'test=', test_file, 'with fraction', percent_dataset, 'had acc', acc)
    return acc


#for each method

writer_cnn = open('outputs_f1/cnn_' + "weda_general" + '_' + get_now_str() + '.txt', 'w')
writer_rnn = open('outputs_f1/rnn_' + "weda_general" + '_' + get_now_str() + '.txt', 'w')


#for each size dataset
for size_folder in size_folders:

    writer_cnn.write(size_folder + '\n')
    writer_rnn.write(size_folder + '\n')

    #get all six datasets
    dataset_folders = [size_folder + '/' + s for s in datasets]

    #for storing the performances
    performances_rnn = {alpha:[] for alpha in alphas}
    performances_cnn = {alpha:[] for alpha in alphas}

    #for each dataset
    for i in range(len(dataset_folders)):

        #initialize all the variables
        dataset_folder = dataset_folders[i]
        dataset = datasets[i]
        num_classes = num_classes_list[i]
        input_size = input_size_list[i]
        word2vec_pickle = dataset_folder + '/word2vec.p'
        word2vec = load_pickle(word2vec_pickle)

        #test each alpha value
        for alpha in alphas:

            train_path = dataset_folder + '/train_weda_' + str(alpha) + '.txt'
            test_path = 'size_data_f1/test/' + dataset + '/test.txt'
            acc = run_cnn(train_path, test_path, num_classes, percent_dataset=1)
            print("cnn aug ", acc, alpha, dataset_folder)

            performances_cnn[alpha].append(acc)
            """
            acc = run_rnn(train_path, test_path, num_classes, percent_dataset=1)
            performances_rnn[alpha].append(acc)
            print("rnn aug", acc)
            """

    writer_cnn.write(str(performances_cnn) + '\n')
    writer_rnn.write(str(performances_rnn)+"\n")
    for alpha in performances_cnn:
        line = str(alpha) + ' : ' + str(sum(performances_cnn[alpha])/len(performances_cnn[alpha]))
        writer_cnn.write(line + '\n')
        print(line)
    """
    for alpha in performances_rnn:
        line = str(alpha) + ' : ' + str(sum(performances_rnn[alpha])/len(performances_rnn[alpha]))
        writer_rnn.write(line + '\n')
        print(line)
    """
    print(performances_cnn)
    print(performances_rnn)

writer_cnn.close()
writer_rnn.close()
    
## TODO: add training without augmentation



writer_cnn = open('outputs_f1/cnn_no_aug_general' + get_now_str() + '.txt', 'w')
writer_rnn = open('outputs_f1/rnn_no_aug_general' + get_now_str() + '.txt', 'w')


#for each size dataset
for size_folder in size_folders:

    writer_cnn.write(size_folder + '\n')
    writer_rnn.write(size_folder + '\n')

    #get all six datasets
    dataset_folders = [size_folder + '/' + s for s in datasets]

    #for storing the performances
    performances_rnn = []
    performances_cnn = []

    #for each dataset
    for i in range(len(dataset_folders)):

        #initialize all the variables
        dataset_folder = dataset_folders[i]
        dataset = datasets[i]
        num_classes = num_classes_list[i]
        input_size = input_size_list[i]
        word2vec_pickle = dataset_folder + '/word2vec.p'
        word2vec = load_pickle(word2vec_pickle)

        train_path = dataset_folder + '/train_orig.txt'
        test_path = 'size_data_f1/test/' + dataset + '/test.txt'
        
        acc = run_cnn(train_path, test_path, num_classes, percent_dataset=1)
        print("cnn", acc)
        performances_cnn.append(acc)

        """
        acc = run_rnn(train_path, test_path, num_classes, percent_dataset=1)
        performances_rnn.append(acc)
        print("rnn", acc)
        """

    writer_cnn.write(str(performances_cnn) + '\n')
    writer_rnn.write(str(performances_rnn)+"\n")

    print(performances_cnn)
    print(performances_rnn)

writer_cnn.close()
writer_rnn.close()


