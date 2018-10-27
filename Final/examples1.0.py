import scipy.misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import preprocessing
from model import Model
from train import predictions, accuracy, load_data

def accuracy_calculate(x,y,L):
    x = preprocessing.get_x("data/train")
    y = preprocessing.get_y1("data/train")
    x = x[y[:,0] == L]
    y = y[y[:,0] == L]
    indices = np.random.choice(len(x), 2000, replace=False)
    examples = x[indices]
    answers = y[indices]

    model = Model()

    with tf.Session(graph=model.graph) as session:
        # Restore model with 82% accuracy on test data
        model.saver.restore(session, "./try2.ckpt")
        log_1, log_2, log_3, log_4, log_5 = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                    model.logits_4, model.logits_5],
                                                    feed_dict={model.x: examples, model.keep_prob: 4.0})
    # Make predictions
    predictions = predictions(log_1, log_2, log_3, log_4, log_5)

    acc = 0.0
    r1 = 0.0
    r2 = 0.0
    for i, el in enumerate(indices):
        number = predictions[i][predictions[i] < 10]
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
    fac = len(x)/100.0
    print('acc:%f'%(acc/fac))
    print('r1:%f'%(r1/fac))
    print('r2:%f'%(r2/fac))

def train(reuse, batch_size=64, number_of_iterations=500):
    """Trains CNN."""
    x_train, y_train, x_test, y_test = load_data()
    x_train[y_train[:,0]==3]
    y_train[y_train[:,0]==3]
    x_test[y_test[:,0]==3]
    y_test[y_test[:,0]==3]
    print("Data uploaded!")
    
    model = Model()
    with tf.Session(graph=model.graph) as session:
        z, n = 0, 0
        # tf.global_variables_initializer().run()

        # Change to True, if you want to restore the model with 82% test accuracy
        if reuse:
            model.saver.restore(session, "./try2.ckpt")
            global_step = tf.trainable_variables()[18]
            # print(session.trainable_variables())
            learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.96)
            var_list = tf.trainable_variables()[8:18]
            for var in var_list:
                session.run(var.initializer)
            model.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(model.loss, var_list=var_list, global_step=global_step)
                
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
                print("Batch accuracy: {},  Loss: {}".format(accuracy(log_1, log_2, log_3, log_4, log_5, y_), l))

        # Evaluate accuracy by parts, if you use GPU and it has low memory.
        # For example, I have 2 GB GPU memory and I need to feed test data by parts(six times by 2178 examples)
        for el in range(6):
            log_1, log_2, log_3, log_4, log_5, y_ = session.run([model.logits_1, model.logits_2, model.logits_3,
                                                                model.logits_4, model.logits_5, model.y],
                                                                feed_dict={model.x: x_test[n:n+2178],
                                                                           model.y: y_test[n:n+2178],
                                                                           model.keep_prob: 1.0})
            n += 2178
            
            # Combine accuracy
            z += accuracy(log_1, log_2, log_3, log_4, log_5, y_)
            
        print("Test accuracy: {}".format((z/6.0)))    

        # Save model in file "try3.ckpt"
        model.saver.save(session, "./try3.ckpt")    

if __name__ == '__main__':
    train(reuse=True)