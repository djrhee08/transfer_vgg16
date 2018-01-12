import numpy as np
import csv

with open('labels') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader]).squeeze()
    labels = labels[:-1]
    print('loaded labels', labels.shape)

    new_labels = []
    for i in range(len(labels)):
        if labels[i][0] == 'daenerys':
            new_labels.append(0)
            print(labels[i][0], new_labels[i])
        elif labels[i][0] == 'jon':
            new_labels.append(1)
            print(labels[i][0], new_labels[i])
        elif labels[i][0] == 'tyrion':
            new_labels.append(2)
            print(labels[i][0], new_labels[i])
        else:
            print("The label", labels[i][0], "is not recognized")


    with open('labels_vecs', 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(new_labels)