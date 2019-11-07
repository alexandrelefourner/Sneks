import types
import numpy as np
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def local_activate(val):
    return 1 if val >= 1 else 0

def test_activation(*answ):
    assert len(answ) == 1, "la fonction activate devrait être l'unique argument ici."
    assert type(answ[0]) == types.FunctionType, "Une fonction est attendue, mais le retour a été  " + str(type(answ[0]))
    
    
    def generate_solution(entry):
        if(entry >= 1):
            return 1
        return 0
    
    params = [0, 0.5, 1, 2, 1.5, -1, (2/3), 0.9]
    
    mistake = False
    for p in params:
        resp = generate_solution(p)
        res = answ[0](p)
        
    
        assert type(res) == type(resp), "Le type de retour attendu était  "+str(type(resp))+", mais a été " + str(type(res))
        
        if(resp != res):
            print("Oups! Votre fonction n'a pas fonctionné correctement dans le cas suivant:\n"+str(p)+" -> "+str(res)+" (au lieu de  "+str(resp)+")")
            mistake = True
            return
            
    if(not mistake):
        print("Parfait!")

        
        

def test_not_gate(*answ):
    assert len(answ) == 1, "la fonction not_gate devrait être l'unique argument ici."
    assert type(answ[0]) == types.FunctionType, "Une fonction est attendue, mais le retour a été  " + str(type(answ[0]))
    
    
    def generate_solution(entry):
        if(entry >= 1):
            return 0
        return 1
    
    params = [0, 0.5, 1, 2, 1.5, -1, (2/3), 0.9]
    
    mistake = False
    for p in params:
        resp = generate_solution(p)
        res = answ[0](p)
        
    
        assert type(res) == type(resp), "Le type de retour attendu était  "+str(type(resp))+", mais a été " + str(type(res))
        
        if(resp != res):
            print("Oups! Votre fonction n'a pas fonctionné correctement dans le cas suivant:\n"+str(p)+" -> "+str(res)+" (au lieu de  "+str(resp)+")")
            mistake = True
            return
            
    if(not mistake):
        print("Parfait!")
def test_or_gate(*answ):
    assert len(answ) == 1, "la fonction or_gate devrait être l'unique argument ici."
    assert type(answ[0]) == types.FunctionType, "Une fonction est attendue, mais le retour a été  " + str(type(answ[0]))
    
    
    def generate_solution(entry_1, entry_2):

        weight_neuron_1 = 1
        weight_neuron_2 = 1

        su = entry_1 * weight_neuron_1 + entry_2 * weight_neuron_2

        return local_activate(su)
    
    params = [(0,0),
                (0,1),
                (1,0),
                (1,1)]
    mistake = False
    for p in params:
        resp = generate_solution(p[0],p[1])
        res = answ[0](p[0],p[1])
        
    
        assert type(res) == type(resp), "Le type de retour attendu était  "+str(type(resp))+", mais a été " + str(type(res))
        
        if(resp != res):
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" (devrait être "+str(resp)+")")
            mistake = True
        else:
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" Ok")
            
    if(not mistake):
        print("Parfait!")
    else:
        print("Oups, ce ne sont pas les bons poids...")

def test_and_gate(*answ):
    assert len(answ) == 1, "la fonction and_gate devrait être l'unique argument ici."
    assert type(answ[0]) == types.FunctionType, "Une fonction est attendue, mais le retour a été  " + str(type(answ[0]))
    
    
    def generate_solution(entry_1, entry_2):

        weight_neuron_1 = 0.5
        weight_neuron_2 = 0.5

        su = entry_1 * weight_neuron_1 + entry_2 * weight_neuron_2

        return local_activate(su)
    
    params = [(0,0),
                (0,1),
                (1,0),
                (1,1)]
    mistake = False
    for p in params:
        resp = generate_solution(p[0],p[1])
        res = answ[0](p[0],p[1])
        
    
        assert type(res) == type(resp), "Le type de retour attendu était  "+str(type(resp))+", mais a été " + str(type(res))
        
        if(resp != res):
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" (devrait être "+str(resp)+")")
            mistake = True
        else:
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" Ok")
            
    if(not mistake):
        print("Parfait!")
    else:
        print("Oups, ce ne sont pas les bons poids...")
        
        
def test_xor_gate(*answ):
    assert len(answ) == 1, "la fonction and_gate devrait être l'unique argument ici."
    assert type(answ[0]) == types.FunctionType, "Une fonction est attendue, mais le retour a été  " + str(type(answ[0]))
    
    
    def generate_solution(entry_1, entry_2):

        su = local_activate(entry_1 + entry_2)-local_activate(entry_1 * 0.5 + entry_2 * 0.5)

        return local_activate(su)
    
    params = [(0,0),
                (0,1),
                (1,0),
                (1,1)]
    mistake = False
    for p in params:
        resp = generate_solution(p[0],p[1])
        res = answ[0](p[0],p[1])
        
    
        assert type(res) == type(resp), "Le type de retour attendu était  "+str(type(resp))+", mais a été " + str(type(res))
        
        if(resp != res):
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" (devrait être "+str(resp)+")")
            mistake = True
        else:
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" Ok")
            
    if(not mistake):
        print("Parfait!")
    else:
        print("Oups, ce ne sont pas les bons poids...")
        
def test_nand_gate(*answ):
    assert len(answ) == 1, "la fonction and_gate devrait être l'unique argument ici."
    assert type(answ[0]) == types.FunctionType, "Une fonction est attendue, mais le retour a été  " + str(type(answ[0]))
    
    
    def generate_solution(entry_1, entry_2):

        return 0 if entry_1 + entry_2 == 2 else 1
    
    params = [(0,0),
                (0,1),
                (1,0),
                (1,1)]
    mistake = False
    for p in params:
        resp = generate_solution(p[0],p[1])
        res = answ[0](p[0],p[1])
        
    
        assert type(res) == type(resp), "Le type de retour attendu était  "+str(type(resp))+", mais a été " + str(type(res))
        
        if(resp != res):
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" (devrait être "+str(resp)+")")
            mistake = True
        else:
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" Ok")
            
    if(not mistake):
        print("Parfait!")
    else:
        print("Oups, ce ne sont pas les bons poids...")

def test_nor_gate(*answ):
    assert len(answ) == 1, "la fonction and_gate devrait être l'unique argument ici."
    assert type(answ[0]) == types.FunctionType, "Une fonction est attendue, mais le retour a été  " + str(type(answ[0]))
    
    
    def generate_solution(entry_1, entry_2):

        return 0 if entry_1 ==1 or entry_2 == 1 else 1
    
    params = [(0,0),
                (0,1),
                (1,0),
                (1,1)]
    mistake = False
    for p in params:
        resp = generate_solution(p[0],p[1])
        res = answ[0](p[0],p[1])
        
    
        assert type(res) == type(resp), "Le type de retour attendu était  "+str(type(resp))+", mais a été " + str(type(res))
        
        if(resp != res):
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" (devrait être "+str(resp)+")")
            mistake = True
        else:
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" Ok")
            
    if(not mistake):
        print("Parfait!")
    else:
        print("Oups, ce ne sont pas les bons poids...")
        

def test_xnor_gate(*answ):
    assert len(answ) == 1, "la fonction and_gate devrait être l'unique argument ici."
    assert type(answ[0]) == types.FunctionType, "Une fonction est attendue, mais le retour a été  " + str(type(answ[0]))
    
    
    def generate_solution(entry_1, entry_2):

        su = local_activate(entry_1 + entry_2)-local_activate(entry_1 * 0.5 + entry_2 * 0.5)

        return 0 if local_activate(su) == 1 else 1
    
    params = [(0,0),
                (0,1),
                (1,0),
                (1,1)]
    mistake = False
    for p in params:
        resp = generate_solution(p[0],p[1])
        res = answ[0](p[0],p[1])
        
    
        assert type(res) == type(resp), "Le type de retour attendu était  "+str(type(resp))+", mais a été " + str(type(res))
        
        if(resp != res):
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" (devrait être "+str(resp)+")")
            mistake = True
        else:
            print(str(p[0])+" & "+str(p[1])+" -> "+str(res)+" Ok")
            
    if(not mistake):
        print("Parfait!")
    else:
        print("Oups, ce ne sont pas les bons poids...")
        
def table_truth(*args):
    network = args[0]
    for i in range(2):
        for j in range(2):
            print(str(i)+" & "+str(j)+" -> "+str(network(i,j)))
            

def test_check_move_the_snake(env, move_function, simulations = 10000):
    rewarded = 0
    avg_movement = 0
    for _ in tqdm(range(simulations)):
        done = False
        mov = 0
        
        observation = env.reset()
        
        while not done:
            mov += 1
            action = move_function(observation)
            observation, reward, done, _ = env.step(action)
        rewarded += 1 if reward > 0 else 0
        avg_movement += mov
            
    print("Réussite dans",str(float('{:,.3f}'.format(rewarded/simulations*100)))+"% des cas (le hasard fait 9,4% en moyenne)")
    print("En moyenne,", float('{:,.3f}'.format(avg_movement/simulations)),"mouvements par épisode")
  
def generate_samples(env, select_movement, episode_qte = 1000, keep_only_correct=False):
    reward = 0
    while(not reward == 1):
        observation = env.reset()
        local_states = []
        local_actions = []
        done = False
        last_was_up = False
        while(not done):
            action = select_movement(observation)
            local_states.append(observation)
            local_actions.append(action)
            observation, reward, done, _ = env.step(action)
    states = np.array(local_states)
    actions = np.array(local_actions)
    
    for i in tqdm(range(episode_qte)):
        observation = env.reset()
        local_states = []
        local_actions = []
        done = False
        last_was_up = False
        while(not done):
            action = select_movement(observation)
            local_states.append(observation)
            local_actions.append(action)
            observation, reward, done, _ = env.step(action)
        if(keep_only_correct and reward <= 0):
            continue
        states = np.vstack((states, local_states))
        local_actions = np.array(local_actions)
        actions = np.concatenate((actions, local_actions))
            
    return states, actions

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    ## Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Label',
           xlabel='Prédiction')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def one_hot_encoder(labels, nb_classes):
    if(type(labels) == list):
        labels = np.array(labels)
    encoder = np.zeros((labels.shape[0], nb_classes))
    encoder[np.arange(labels.shape[0]), labels] = 1
    return encoder
	
def generate_batch(x, y, batch_size=32):
    at_point = 0
    while(at_point < x.shape[0]):
        end_point = min(at_point + batch_size, x.shape[0])
        yield x[at_point:end_point], y[at_point:end_point]
        at_point = end_point