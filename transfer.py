import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import block_diag
from scipy.spatial.distance import pdist, squareform
from orthogonal_matrices import random_orthogonal

######Parameters###################
init_eta = 0.005
eta_decay = 1.0 #multiplicative per eta_decay_epoch epochs
eta_decay_epoch = 10
nepochs = 100000
termination_thresh = 0.01 # stop at this loss
nruns = 500
run_offset = 0
num_hidden = 4
save_detailed = False # currently most useful for 3 layer, saves detailed info
                      # about the evolution of the penultimate weights and reps.
save_summarized_detailed = True # same but saves a less ridiculous amount of data
###################################
nonlinearity_function = tf.nn.leaky_relu
num_inputs_per = 2
num_outputs_per = 1

binary_x_data = np.array([[0,0],
                          [0,1],
                          [1,0],
                          [1,1]])

x_data = block_diag(binary_x_data, binary_x_data)

XOR_data = np.array([[0],
                     [1],
                     [1],
                     [0]])

AND_data = np.array([[0],
                     [0],
                     [0],
                     [1]])

y_data = block_diag(XOR_data, XOR_data)
y_data_no = block_diag(XOR_data, AND_data)

y_datasets = [y_data_no, y_data, y_data_no] # for the third case, just won't train

print(y_data)
print(y_data_no)

np.savetxt("no_analogy_data.csv", y_data_no, delimiter=',')
np.savetxt("analogy_data.csv", y_data, delimiter=',')

num_inputs = 2*num_inputs_per
num_outputs = 2*num_outputs_per
num_examples_per = len(binary_x_data)
num_examples = len(x_data)


for rseed in xrange(run_offset, run_offset + nruns):#[66, 80, 104, 107]: #
    for nonlinear in [True]:
        for nlayer in [3]: #[3]: #
            for analogous in [1, 0, 2]:
                num_hidden = num_hidden
                print "nlayer %i nonlinear %i analogous %i run %i" % (nlayer, nonlinear, analogous, rseed)
                filename_prefix = "XOR_results_2/nlayer_%i_nonlinear_%i_analogous_%i_rseed_%i_" %(nlayer,nonlinear,analogous,rseed)

                np.random.seed(rseed)
                tf.set_random_seed(rseed)
                this_x_data = x_data
                this_y_data = y_datasets[analogous] 

                input_ph = tf.placeholder(tf.float32, shape=[None, num_inputs])
                target_ph = tf.placeholder(tf.float32, shape=[None, num_outputs])


                Win = tf.Variable(tf.random_uniform([num_inputs,num_hidden],0.,0.5/(num_hidden+num_inputs)))
                bi = tf.Variable(tf.zeros([num_hidden]))
                internal_rep = tf.matmul(input_ph, Win) + bi
                hidden_weights = []
                if nonlinear:
                    internal_rep = nonlinearity_function(internal_rep)

                for layer_i in range(1, nlayer-1):
                    if layer_i == nlayer-2:
                        penultimate_rep = internal_rep
                    W = tf.Variable(tf.random_normal([num_hidden,num_hidden],0.,0.5/num_hidden))
                    b = tf.Variable(tf.zeros([num_hidden]))
                    hidden_weights.append((W, b))
                    internal_rep = tf.matmul(internal_rep, W) + b
                    if nonlinear:
                        internal_rep = nonlinearity_function(internal_rep)

                bo = tf.Variable(tf.zeros([num_outputs]))
                Wout = tf.Variable(tf.random_uniform([num_hidden,num_outputs],0.,0.5/(num_hidden+num_outputs)))
                pre_output = tf.matmul(internal_rep, Wout) + bo

                if nonlinear:
                    output = nonlinearity_function(pre_output)
                else:
                    output = pre_output

                d1_loss = tf.reduce_sum(tf.square(output[:num_examples_per, :num_outputs_per] - target_ph[:num_examples_per, :num_outputs_per]))
                d2_loss = tf.reduce_sum(tf.square(output[num_examples_per:, num_outputs_per:] - target_ph[num_examples_per:, num_outputs_per:]))
                loss = d1_loss + d2_loss
                output_grad = tf.gradients(loss,[output])[0]
                eta_ph = tf.placeholder(tf.float32)
                optimizer = tf.train.GradientDescentOptimizer(eta_ph)
                if analogous < 2:
                    train = optimizer.minimize(loss)
                else:
                    train = optimizer.minimize(d1_loss) # only train on one task

                init = tf.global_variables_initializer()

                sess = tf.Session()
                sess.run(init)

                def test_accuracy():
                    MSE = sess.run(loss,
                                   feed_dict={input_ph: this_x_data,target_ph: this_y_data})
                    MSE /= num_examples 

                    d1_MSE = sess.run(d1_loss,
                                   feed_dict={input_ph: this_x_data,target_ph: this_y_data})
                    d1_MSE /= num_examples_per 
                    return MSE, d1_MSE


                def get_outputs():
                    return sess.run(output,feed_dict={input_ph: this_x_data})


                def print_preoutputs():
                    print sess.run(pre_output,feed_dict={input_ph: this_x_data})


                def save_activations(tf_object,filename,remove_old=True):
                    if remove_old and os.path.exists(filename):
                        os.remove(filename)
                    with open(filename,'ab') as fout:
                        res = sess.run(tf_object, feed_dict={input_ph: this_x_data})
                        np.savetxt(fout, res, delimiter=',')


                def get_activations(tf_object):
                    return sess.run(tf_object, feed_dict={input_ph: this_x_data})


                def save_weights(tf_object,filename,remove_old=True):
                    if remove_old and os.path.exists(filename):
                        os.remove(filename)
                    with open(filename,'ab') as fout:
                        np.savetxt(fout,sess.run(tf_object),delimiter=',')


                def run_train_epoch():
                    sess.run(train,feed_dict={eta_ph: curr_eta,input_ph: this_x_data,target_ph: this_y_data})


                def save_penultimate_details(filename_prefix, epoch):
                    # save activations for the different inputs
                    save_activations(penultimate_rep, filename_prefix + "penultimate_hidden_reps_epoch_%i.csv" % epoch)
                    W, b = hidden_weights[-1]
                    U, S, V = np.linalg.svd(sess.run(W), full_matrices=False)
                    np.savetxt(filename_prefix + "penultimate_U_epoch_%i.csv" % epoch, U, delimiter=',') 
                    np.savetxt(filename_prefix + "penultimate_S_epoch_%i.csv" % epoch, S, delimiter=',') 
                    np.savetxt(filename_prefix + "penultimate_V_epoch_%i.csv" % epoch, V, delimiter=',') 


                def save_summarized_penultimate_details(outfiles, epoch):
                    # save activations for the different inputs
                    reps = get_activations(penultimate_rep)
                    W, b = hidden_weights[-1]
                    U, S, _ = np.linalg.svd(sess.run(W), full_matrices=False)
                    simils = squareform(pdist(reps, metric='cosine')) 
                    reps /= np.sqrt(np.sum(np.square(reps), axis=-1, keepdims=True))
                    projs = np.matmul(reps, U)
                    sout, svout, pout = outfiles
                    for i in range(num_examples-1):
                        for j in range(i+1, num_examples):
                            sout.write("%i, %i, %i, %f\n" % (epoch, i, j, simils[i, j]))

                    for i in range(len(S)):
                        svout.write("%i, %i, %f\n" % (epoch, i, S[i]))

                    for i in range(num_examples):
                        for j in range(num_hidden):
                            pout.write("%i, %i, %i, %f\n" % (epoch, i, j, projs[i, j]))


                print "Initial MSE: %f, %f" %(test_accuracy())
                #loaded_pre_outputs = np.loadtxt(pre_output_filename_to_load,delimiter=',')
                if save_summarized_detailed:
                    simil_filename = filename_prefix + "penultimate_simil_track.csv"
                    sout = open(simil_filename, "w") 
                    sout.write("epoch, rep_i, rep_j, cosine_similarity\n") # note: this should actually be cosine _distance_, whoops
                    singular_value_filename = filename_prefix + "penultimate_S_track.csv"
                    svout = open(singular_value_filename, "w") 
                    svout.write("epoch, rank, S\n")
                    proj_filename = filename_prefix + "penultimate_proj_track.csv"
                    pout = open(proj_filename, "w") 
                    pout.write("epoch, rep_i, mode_j, projection\n")
                    outfiles = (sout, svout, pout)

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
                            if curr_mse[1] < termination_thresh:
                                print("Early stop!")
                                break
#                            if epoch % 100 == 0:
#                                save_activations(internal_rep,filename_prefix+"epoch_%i_internal_rep.csv" %epoch)
#                                save_activations(pre_output,filename_prefix+"epoch_%i_pre_outputs.csv" %epoch)
                        if save_detailed and epoch % 100 == 0:
                            save_penultimate_details(filename_prefix, epoch)

                        if save_summarized_detailed and epoch % 100 == 0:
                            save_summarized_penultimate_details(outfiles, epoch)

                        
                        if epoch % eta_decay_epoch == 0:
                            curr_eta *= eta_decay
                    
                if save_summarized_detailed:
                    for i in range(len(outfiles)):
                        outfiles[i].close()

                print "Final MSE: %f, %f" %(test_accuracy())
                tf.reset_default_graph()
