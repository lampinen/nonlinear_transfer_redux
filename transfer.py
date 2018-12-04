import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import block_diag
from orthogonal_matrices import random_orthogonal

######Parameters###################
init_eta = 0.005
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 200000
termination_thresh = 0.01 # stop at this loss
nruns = 200
run_offset = 0
num_inputs = 6
num_outputs = 8
num_hidden = 6
###################################
nonlinearity_function = tf.nn.relu

sigma_31 = np.array(
    [[1, 1, 0, 0, 0, 0, 0, 0],
     [1, 0, 1, 0, 0, 0, 0, 0],
     [1, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 0, 1, 0],
     [0, 0, 0, 0, 1, 0, 0, 1]])

rt_3 = np.sqrt(3)
rt_2 = np.sqrt(2)
sigma_31_no = np.array(
    [[1, 1, 0, 0, 0, 0, 0, 0],
     [1, 0, 1, 0, 0, 0, 0, 0],
     [1, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 2, 0, 0, 0],
     [0, 0, 0, 0, 0, 1/rt_2, 1/rt_2, 0],
     [0, 0, 0, 0, 0, 0, 0, 1]])

for rseed in xrange(run_offset, run_offset + nruns):
    np.random.seed(rseed)

    _, S1, V1 = np.linalg.svd(sigma_31[:num_inputs//2, :num_outputs//2], full_matrices=False)
    struct = random_orthogonal(num_inputs//2)
    U = block_diag(struct, struct)
    S = block_diag(np.diag(S1), np.diag(S1))
    V = block_diag(V1, V1)
    sigma_31 = np.matmul(U, np.matmul(S, V))
    _, S, V = np.linalg.svd(sigma_31, full_matrices=False)
    print(sigma_31)
    print(S)

#    _, S1, V1 = np.linalg.svd(sigma_31_no[:num_inputs//2, :num_outputs//2], full_matrices=False)
    _, S2, V2 = np.linalg.svd(sigma_31_no[num_inputs//2:, num_outputs//2:], full_matrices=False)
    U = block_diag(struct, random_orthogonal(num_inputs//2)) 
    S = block_diag(np.diag(S1), np.diag(S2))
    V = block_diag(V1, V2)
    sigma_31_no = np.matmul(U, np.matmul(S, V))
    _, S, V = np.linalg.svd(sigma_31_no, full_matrices=False)
    print()
    print(sigma_31_no)
    print(S)


    x_struct = random_orthogonal(num_inputs//2)
    x_data = block_diag(x_struct, x_struct) 
    print(x_data)

    y_data = np.matmul(x_data.transpose(), sigma_31)
    y_data_no = np.matmul(x_data.transpose(), sigma_31_no)

    y_datasets = [y_data, y_data_no]

    print(y_data)
    print(y_data_no)
    if rseed == 0:
        np.savetxt("no_analogy_data.csv", y_data_no, delimiter=',')
        np.savetxt("analogy_data.csv", y_data, delimiter=',')

    for nonlinear in [True, False]:
        nonlinearity_function = tf.nn.leaky_relu
        for nlayer in [4, 3, 2]:
            for analogous in [0, 1]:
                num_hidden = num_hidden
                print "nlayer %i nonlinear %i analogous %i run %i" % (nlayer, nonlinear, analogous, rseed)
                filename_prefix = "results/nlayer_%i_nonlinear_%i_analogous_%i_rseed_%i_" %(nlayer,nonlinear,analogous,rseed)

                np.random.seed(rseed)
                tf.set_random_seed(rseed)
                this_x_data = x_data
                this_y_data = y_datasets[analogous] 

                input_ph = tf.placeholder(tf.float32, shape=[None, num_inputs])
                target_ph = tf.placeholder(tf.float32, shape=[None, num_outputs])


                Win = tf.Variable(tf.random_uniform([num_hidden,num_inputs],0.,0.5/(num_hidden+num_inputs)))
                bi = tf.Variable(tf.ones([num_hidden,]))
                internal_rep = tf.matmul(Win, input_ph) + bi
                hidden_weights = []
                if nonlinear:
                    internal_rep = nonlinearity_function(internal_rep)

                for layer_i in range(1, nlayer-1):
                    W = tf.Variable(tf.random_normal([num_hidden,num_hidden],0.,0.5/num_hidden))
                    b = tf.Variable(tf.ones([num_hidden,]))
                    internal_rep = tf.matmul(W, internal_rep) + b
                    if nonlinear:
                        internal_rep = nonlinearity_function(internal_rep)

                bo = tf.Variable(tf.ones([num_outputs,1]))
                Wout = tf.Variable(tf.random_uniform([num_outputs,num_hidden],0.,0.5/(num_hidden+num_outputs)))
                pre_output = tf.matmul(Wout, internal_rep) + bo

                if nonlinear:
                    output = nonlinearity_function(pre_output)
                else:
                    output = pre_output

                loss = tf.reduce_sum(tf.square(output - tf.transpose(target_ph)))# +0.05*(tf.nn.l2_loss(internal_rep))
                d1_loss = tf.reduce_sum(tf.square(output[:4, :3] - tf.transpose(target_ph)[:4, :3]))
                output_grad = tf.gradients(loss,[output])[0]
                eta_ph = tf.placeholder(tf.float32)
                optimizer = tf.train.GradientDescentOptimizer(eta_ph)
                train = optimizer.minimize(loss)

                init = tf.global_variables_initializer()

                sess = tf.Session()
                sess.run(init)

                def test_accuracy():
                    MSE = sess.run(loss,
                                   feed_dict={input_ph: this_x_data,target_ph: this_y_data})
                    MSE /= num_inputs 

                    d1_MSE = sess.run(d1_loss,
                                   feed_dict={input_ph: this_x_data,target_ph: this_y_data})
                    d1_MSE /= num_inputs 
                    return MSE, d1_MSE

                def print_outputs():
                    print sess.run(output,feed_dict={input_ph: this_x_data})


                def print_preoutputs():
                    print sess.run(pre_output,feed_dict={input_ph: this_x_data})


                def save_activations(tf_object,filename,remove_old=True):
                    if remove_old and os.path.exists(filename):
                        os.remove(filename)
                    with open(filename,'ab') as fout:
                        res = sess.run(tf_object, feed_dict={input_ph: this_x_data})
                        np.savetxt(fout, res, delimiter=',')


                def save_weights(tf_object,filename,remove_old=True):
                    if remove_old and os.path.exists(filename):
                        os.remove(filename)
                    with open(filename,'ab') as fout:
                        np.savetxt(fout,sess.run(tf_object),delimiter=',')


                def run_train_epoch():
                    sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: this_x_data,target_ph: this_y_data})

                print "Initial MSE: %f, %f" %(test_accuracy())

                #loaded_pre_outputs = np.loadtxt(pre_output_filename_to_load,delimiter=',')

                curr_eta = init_eta
                rep_track = []
                loss_filename = filename_prefix + "loss_track.csv"
                with open(loss_filename, 'w') as fout:
                    fout.write("epoch, MSE, d1_MSE\n")
                    curr_mse = test_accuracy()
                    fout.write("%i, %f, %f\n" %(0, curr_mse[0], curr_mse[1]))
                    for epoch in xrange(nepochs):
                        run_train_epoch()
                        if epoch % 5 == 0:
                            curr_mse = test_accuracy()
                            print "epoch: %i, MSEs: %f, %f" %(epoch, curr_mse[0], curr_mse[1])	
                            fout.write("%i, %f, %f\n" %(epoch, curr_mse[0], curr_mse[1]))
                            if curr_mse[0] < termination_thresh:
                                print("Early stop!")
                                break
#                            if epoch % 100 == 0:
#                                save_activations(internal_rep,filename_prefix+"epoch_%i_internal_rep.csv" %epoch)
#                                save_activations(pre_output,filename_prefix+"epoch_%i_pre_outputs.csv" %epoch)
                        
                        if epoch % eta_decay_epoch == 0:
                            curr_eta *= eta_decay
                    
                print "Final MSE: %f, %f" %(test_accuracy())
                tf.reset_default_graph()
