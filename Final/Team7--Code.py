import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import preprocessing
from model import Model
from train import predictions, accuracy, load_data
"""
m_accuracy(L, log_1, log_2, log_3, log_4, log_5, y_):
calculate the accuracy of prediction

L is the length of number
log_1~5 is a tensor containing the probability of each digit(0~10, 10 means no digit)


ckpt_accuracy(L,ckpt,cut=True):
calculate the accuracy of trained ckpt

L is the length of number of test data
if cut=True, it will ignore the digit longer than L when calculate accuracy

return acc, r1, r2
r1 is the error of length of number
r2 is the error of wrong prediction of digit


train
"""


#modify accuracy function, could cut the rest part of digits
def m_accuracy(L, log_1, log_2, log_3, log_4, log_5, y_):
    #if L==0, don't cut anything
    if L==0:
        return accuracy(log_1, log_2, log_3, log_4, log_5, y_)

    #cut the digit longer than L
    preds = predictions(log_1, log_2, log_3, log_4, log_5)
    preds = preds[:,:L]

    acc = 0.0
    for i in range(len(preds)):
        number = preds[i][preds[i] < 10]
        ans = y_[i][y_[i] < 10]

        if len(ans[1:])!=len(number):
            continue

        if not (ans[1:]-number).any():
            acc += 1.0
        else:
            try:
                ans = int(''.join(ans[1:]))
                number = int(''.join(number))
                if ans == number:
                    acc += 1.0
            except:
                pass
    fac = len(preds)/100.0
    return acc/fac
    
#calculate accuracy of the trained ckpt
def ckpt_accuracy(L,ckpt,cut=True):
    x = preprocessing.get_x("data/test")
    y = preprocessing.get_y1("data/test")
    x = x[y[:,0] == L]
    y = y[y[:,0] == L]
    
    if len(x)>2000:
        indices = np.random.choice(len(x), 2000, replace=False)
        examples = x[indices]
        answers = y[indices]

    else :
        examples = x
        answers = y       

    model = Model()

    with tf.Session(graph=model.graph) as session:
        # Restore model with filename as ckpt
        model.saver.restore(session, "./"+ckpt)
        log_1, log_2, log_3, log_4, log_5 = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                    model.logits_4, model.logits_5],
                                                    feed_dict={model.x: examples, model.keep_prob: 4.0})
    # Make predictions
    preds = predictions(log_1, log_2, log_3, log_4, log_5)

    #cut the digit longer than L
    if cut:
        preds = preds[:,:L]

    #r1 is first type error that predict wrong length
    #r2 is second type error that predict wrong number
    acc = 0.0
    r1 = 0.0
    r2 = 0.0
    for i, el in enumerate(indices):
        number = preds[i][preds[i] < 10]
        ans = answers[i][answers[i] < 10]

        if len(ans[1:])!=len(number):
            r1 += 1.0
            continue
        if not (ans[1:]-number).any():
            acc += 1.0
        else:
            r2 += 1.0
            try:
                ans = int(''.join(ans[1:]))
                number = int(''.join(number))
                if ans == number:
                    acc += 1.0
            except:
                pass
    fac = len(examples)/100.0
    print('ckpt_acc:%f'%(acc/fac))
    print('r1:%f'%(r1/fac))
    print('r2:%f'%(r2/fac))
    return acc, r1, r2

def train(L=0, batch_size=64, number_of_iterations=2000):
    #load data and select data with particular length(optional)
    x_train, y_train, x_test, y_test = load_data()
    if L!=0:
        x_train = x_train[y_train[:,0]==L]
        y_train = y_train[y_train[:,0]==L]
        x_test = x_test[y_test[:,0]==L]
        y_test = y_test[y_test[:,0]==L]
    print("Data uploaded!")
    
    model = Model()
    with tf.Session(graph=model.graph) as session:
        z, n = 0, 0
        model.saver.restore(session, "./try2.ckpt")#or you can base any ckpt
        global_step = tf.trainable_variables()[18]
        learning_rate = tf.train.exponential_decay(0.01, global_step, 10000, 0.96)
        optimizer = tf.train.AdagradOptimizer(learning_rate)

        var_list = tf.trainable_variables()[8:18]
        tf.variables_initializer(var_list).run()
        if L>0:
            var_list = var_list[0:L]+var_list[5:5+L]


        model.optimizer = optimizer.minimize(model.loss, var_list=var_list, global_step=global_step)           
        tf.variables_initializer(optimizer.variables()).run() 

        for i in range(number_of_iterations):
            indices = np.random.choice(len(y_train), batch_size, replace=False)
            bat_x, bat_y = x_train[indices], y_train[indices]
            _, l = session.run([model.optimizer, model.loss], feed_dict={model.x: bat_x, model.y: bat_y,
                                                                         model.keep_prob: 0.5})
            # Check batch accuracy and loss
            if i % 100 == 0:
                log_1, log_2, log_3, log_4, log_5, y_ = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                                    model.logits_4, model.logits_5, model.y],
                                                                    feed_dict={model.x: bat_x, model.y: bat_y,
                                                                               model.keep_prob: 1.0})
                print("Iteration number: {}".format(i))
                print("Batch accuracy: {},  Loss: {}".format(m_accuracy(L, log_1, log_2, log_3, log_4, log_5, y_), l))

        # Evaluate accuracy by parts, if you use GPU and it has low memory.        
        if len(x_test)<2178:
            x_test_partial = x_test
            y_test_partial = y_test
        else:
            x_test_partial = x_test[:2178]
            y_test_partial = y_test[:2178]
        log_1, log_2, log_3, log_4, log_5, y_ = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                            model.logits_4, model.logits_5, model.y],
                                                            feed_dict={model.x: x_test_partial,
                                                                       model.y: y_test_partial,
                                                                       model.keep_prob: 1.0})

        # Combine accuracy
        z += m_accuracy(L, log_1, log_2, log_3, log_4, log_5, y_)
            
        print("Test accuracy: {}".format((z)))    

        # Save model
        model.saver.save(session, "./try8.ckpt")

if __name__ == '__main__':
    train(L=3)
    ckpt_accuracy(L=3, ckpt="try8.ckpt")#you can change ckpt to which you want to test
