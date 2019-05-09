import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import block_diag
from scipy.spatial.distance import pdist, squareform
from orthogonal_matrices import random_orthogonal

######Parameters###################
init_eta = 0.01
weight_size = 1e-3
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 50000
termination_thresh = 0.01 # stop at this loss
nruns = 100
run_offset = 0
num_hidden_rep = 16 
num_hidden_shared = 32
save_detailed = False # currently most useful for 3 layer, saves detailed info
                      # about the evolution of the penultimate weights and reps.
save_summarized_detailed = True # same but saves a less ridiculous amount of data
###################################
nonlinearity_function = tf.nn.sigmoid
num_inputs_per = 2
num_outputs_per = 1


x_data = np.loadtxt('rogers_inputs.csv', delimiter=',')
y_data = np.loadtxt('rogers_outputs.csv', delimiter=',')

num_inputs = 48 
num_outputs = 224 
num_examples = len(x_data)


for rseed in xrange(run_offset, run_offset + nruns):#[66, 80, 104, 107]: #
    for nonlinear in [True]:
        print "nonlinear %i run %i" % (nonlinear, rseed)
        filename_prefix = "rogers_results/nonlinear_%i_rseed_%i_" %(nonlinear,rseed)

        np.random.seed(rseed)
        tf.set_random_seed(rseed)

        input_ph = tf.placeholder(tf.float32, shape=[None, num_inputs])
        target_ph = tf.placeholder(tf.float32, shape=[None, num_outputs])

        item_input = input_ph[:, :32]
        context_input = input_ph[:, 32:]

        Witem = tf.Variable(tf.random_uniform([32, num_hidden_rep], -weight_size, weight_size))
        bitem = tf.Variable(tf.zeros([num_hidden_rep]))
        Wctxt = tf.Variable(tf.random_uniform([16, num_hidden_rep], -weight_size, weight_size))
        bctxt = tf.Variable(tf.zeros([num_hidden_rep]))

        item_rep = tf.matmul(item_input, Witem) + bitem
        if nonlinear:
            item_rep = nonlinearity_function(item_rep)
        ctxt_rep = tf.matmul(context_input, Wctxt) + bctxt
        if nonlinear:
            ctxt_rep = nonlinearity_function(ctxt_rep)
 
        combined_rep = tf.concat([item_rep, ctxt_rep], axis=-1)
        W = tf.Variable(tf.random_uniform([2*num_hidden_rep,num_hidden_shared],-weight_size,weight_size))
        b = tf.Variable(tf.zeros([num_hidden_shared]))
        rep_2 = tf.matmul(combined_rep, W) + b 
        if nonlinear:
            rep_2 = nonlinearity_function(rep_2)

        bo = tf.Variable(tf.zeros([num_outputs]))
        Wout = tf.Variable(tf.random_uniform([num_hidden_shared,num_outputs],-weight_size, weight_size))
        pre_output = tf.matmul(rep_2, Wout) + bo
        if nonlinear:
            output = nonlinearity_function(pre_output)
        else:
            output = pre_output

        loss = tf.reduce_sum(tf.square(output - target_ph))
        output_grad = tf.gradients(loss,[output])[0]
        eta_ph = tf.placeholder(tf.float32)
        optimizer = tf.train.GradientDescentOptimizer(eta_ph)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        def test_accuracy():
            MSE = sess.run(loss,
                           feed_dict={input_ph: x_data,target_ph: y_data})
            MSE /= num_examples 
            return MSE


        def get_outputs():
            return sess.run(output,feed_dict={input_ph: x_data})


        def print_preoutputs():
            print sess.run(pre_output,feed_dict={input_ph: x_data})


        def save_activations(tf_object,filename,remove_old=True):
            if remove_old and os.path.exists(filename):
                os.remove(filename)
            with open(filename,'ab') as fout:
                res = sess.run(tf_object, feed_dict={input_ph: x_data})
                np.savetxt(fout, res, delimiter=',')


        def get_activations(tf_object):
            return sess.run(tf_object, feed_dict={input_ph: x_data})


        def save_weights(tf_object,filename,remove_old=True):
            if remove_old and os.path.exists(filename):
                os.remove(filename)
            with open(filename,'ab') as fout:
                np.savetxt(fout,sess.run(tf_object),delimiter=',')


        def run_train_epoch():
            sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: x_data,target_ph: y_data})
                    

        def save_item_RSA(filename_prefix, epoch):
            reps = get_activations(item_rep)
            RSA = squareform(pdist(reps, metric='euclidean'))
            filename = filename_prefix + "epoch_%i_item_RSA.csv"
            with open(filename,'w') as fout:
                np.savetxt(fout,RSA,delimiter=',')


        print "Initial MSE: %f" %(test_accuracy())
        #loaded_pre_outputs = np.loadtxt(pre_output_filename_to_load,delimiter=',')

        curr_eta = init_eta
        rep_track = []
        loss_filename = filename_prefix + "loss_track.csv"
        with open(loss_filename, 'w') as fout:
            fout.write("epoch, MSE, d1_MSE\n")
            curr_mse = test_accuracy()
            fout.write("%i, %f\n" %(0, curr_mse))
            for epoch in xrange(nepochs):
                run_train_epoch()
                if epoch % 5 == 0:
                    curr_mse = test_accuracy()
                    print "epoch: %i, MSEs: %f" %(epoch, curr_mse)	
                    fout.write("%i, %f\n" %(epoch, curr_mse))
                    if curr_mse < termination_thresh:
                        print("Early stop!")
                        break
#                            if epoch % 100 == 0:
#                                save_activations(internal_rep,filename_prefix+"epoch_%i_internal_rep.csv" %epoch)
#                                save_activations(pre_output,filename_prefix+"epoch_%i_pre_outputs.csv" %epoch)
                
                if epoch % eta_decay_epoch == 0:
                    curr_eta *= eta_decay
            
        if save_summarized_detailed:
            save_item_RSA(filename_prefix, epoch)

        print "Final MSE: %f" %(test_accuracy())
        tf.reset_default_graph()
