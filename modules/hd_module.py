import random
import numpy as np
import os
import pickle

from scipy.special import softmax

class hd_module:
    def __init__(self):
        # HD dimension used
        self.dim = 10000
        self.num_sensors = 7
        self.num_actuators = 4
        self.sensor_weight = 1
        self.threshold_known = 0.08
        self.softmax_param = 7.79

        # method 1, 2, 3
        self.encoding_method = self.encodeing_method_helper(3)

        self.output_vectors = []
        self.output_actuators = []

        self.outdir = './data/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        sensor_ids_fname = self.outdir + 'hd_sensor_ids_dim_' + str(self.dim) + '.npy'
        sensor_vals_fname = self.outdir + 'hd_sensor_vals_dim_' + str(self.dim) + '.npy'
        sensor_dist_fname = self.outdir + 'hd_sensor_dist_' + str(self.dim) + '.npy'
        sensor_last_fname= self.outdir + 'hd_sensor_last_' + str(self.dim) + '.npy'
        actuator_vals_fname = self.outdir + 'hd_actuator_vals_dim_' + str(self.dim) + '.npy'

        # Load/create HD items
        if os.path.exists(sensor_ids_fname):
            self.hd_sensor_ids = np.load(sensor_ids_fname)
        else:
            print("creating sensor id mem")
            self.hd_sensor_ids = self.create_bipolar_mem(self.num_sensors,self.dim)
            np.save(sensor_ids_fname, self.hd_sensor_ids)

        if os.path.exists(sensor_vals_fname):
            self.hd_sensor_vals = np.load(sensor_vals_fname)
        else:
            print("creating sensor val mem")
            self.hd_sensor_vals = self.create_bipolar_mem(2,self.dim)
            np.save(sensor_vals_fname, self.hd_sensor_vals)

        if os.path.exists(sensor_dist_fname):
            self.hd_sensor_dist = np.load(sensor_dist_fname)
        else:
            print("creating sensor dist mem")
            #self.hd_sensor_dist = self.create_bipolar_CIM(19,self.dim)
            self.hd_sensor_dist = self.create_bipolar_mem(3,self.dim)
            np.save(sensor_dist_fname, self.hd_sensor_dist)

        if os.path.exists(sensor_last_fname):
            self.hd_sensor_last = np.load(sensor_last_fname)
        else:
            print("creating sensor last mem")
            self.hd_sensor_last = self.create_bipolar_mem(4,self.dim)
            np.save(sensor_last_fname, self.hd_sensor_last)

        if os.path.exists(actuator_vals_fname):
            self.hd_actuator_vals = np.load(actuator_vals_fname)
        else:
            print("creating actuator val mem")
            self.hd_actuator_vals = self.create_bipolar_mem(self.num_actuators,self.dim)
            np.save(actuator_vals_fname, self.hd_actuator_vals)

        # Initialize program vector
        self.hd_program_vec = np.zeros((self.dim,), dtype = np.int)

        # Initialize condition vector
        self.hd_cond_vec = np.zeros((self.dim,), dtype = np.int)
        self.num_cond = 0
        self.num_thrown = 0


    def create_bipolar_mem(self, numitem, dim):
        # Creates random bipolar memory of given size

        rand_arr = np.rint(np.random.rand(numitem, dim)).astype(np.int8)
        return (rand_arr*2 - 1)

    def create_bipolar_CIM(self, numitem, dim):
        # Creates random bipolar memory of given size
        rand_arr = np.rint(np.random.rand(dim,)).astype(np.int8)
        bipolar_arr = rand_arr*2 - 1
        neg_arr = -bipolar_arr

        CIM = np.zeros((numitem,dim)).astype(np.int8)
        block = dim//numitem
        for i in range(numitem):
            CIM[i,:i*block] = bipolar_arr[:i*block]
            CIM[i,i*block:] = neg_arr[i*block:]
        '''
        rand_arr = np.rint(np.random.rand(2, dim)).astype(np.int8)
        bipolar_arr = rand_arr*2 - 1

        CIM = np.zeros((numitem,dim)).astype(np.int8)
        block = dim//numitem
        for i in range(numitem):
            CIM[i,:i*block] = bipolar_arr[0,:i*block]
            CIM[i,i*block:] = bipolar_arr[1,i*block:]
        '''

        return CIM

    def hd_mul(self, A, B):
        # Return element-wise multiplication between bipolar HD vectors
        # inputs:
        #   - A: bipolar HD vector
        #   - B: bipolar HD vector
        # outputs:
        #   - A*B: bipolar HD vector
        return np.multiply(A,B,dtype = np.int8)

    def hd_perm(self, A):
        # Return right cyclic shift of input bipolar vector
        # inputs:
        #   - A: bipiolar HD vector
        # outputs:
        #   - rho(A): bipolar HD vector
        return np.roll(A,1)

    def hd_threshold(self, A):
        # Given integer vector, threshold at zero to bipolar
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - [A]: bipolar HD vector
        #return (np.greater_equal(A,0, dtype=np.int8)*2-1)
        return (np.int8(np.greater_equal(A,0))*2-1)


    def search_actuator_vals(self, A):
        # Find the nearest item in 'hd_actuator_vals' according to Hamming distance
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - i: integer index of closest item in 'hd_actuator_vals'
        dists = np.matmul(self.hd_actuator_vals, A, dtype = np.int)
        return np.argmax(dists)

    def softmax_actuator_vals(self, A, softmax_param):
        # Find the nearest item in 'hd_actuator_vals' according to Hamming distance
        # inputs:
        #   - A: bipolar HD vector
        # outputs:
        #   - i: integer index of closest item in 'hd_actuator_vals'
        dists = np.matmul(self.hd_actuator_vals, A, dtype = np.int)
        probs = softmax(dists/np.max(dists)*softmax_param)
        #print(dists)
        #print(probs)

        return np.random.choice(4, p = probs)

    def encodeing_method_helper(self, method):
        if method==1:
            return self.encode_sensors
        return lambda sensor_in, train: self.encode_sensors_directional(sensor_in, train, method)

        #method 1
    def encode_sensors(self, sensor_in, train):
        # Encode sensory data into HD space
        # Currently binds together all sensor inputs
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector
        sensor_vec = np.ones((self.dim,), dtype = np.int8) #bind
        #sensor_vec = np.zeros((self.dim,), dtype = np.int8) #bundle
        for i,sensor_val in enumerate(sensor_in[:4]):
            permuted_vec = self.hd_sensor_vals[sensor_val,:]
            for j in range(i):
                # permute hd_sensor_val based on the corresponding sensor id
                permuted_vec = self.hd_perm(permuted_vec)
            binded_sensor = self.hd_mul(self.hd_sensor_ids[i,:],permuted_vec)
            sensor_vec = self.hd_mul(sensor_vec, binded_sensor) #bind
            #sensor_vec = sensor_vec + binded_sensor #bundle
        #sensor_vec = self.hd_threshold(sensor_vec) #bundle

        #xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], self.hd_sensor_dist[sensor_in[4] + 9,:])
        #ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], self.hd_perm(self.hd_sensor_dist[sensor_in[5] + 9,:]))
        if sensor_in[4] > 0:
            xval = self.hd_sensor_dist[2]
        elif sensor_in[4] < 0:
            xval = self.hd_sensor_dist[0]
        else:
            xval = self.hd_sensor_dist[1]

        if sensor_in[5] > 0:
            yval = self.hd_sensor_dist[2]
        elif sensor_in[5] < 0:
            yval = self.hd_sensor_dist[0]
        else:
            yval = self.hd_sensor_dist[1]
        yval = self.hd_perm(yval)

        xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], xval)
        ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], yval)
        dist_vec = self.hd_mul(xdist_vec, ydist_vec)

        last_vec = self.hd_sensor_last[sensor_in[6],:]
        #last_vec = self.hd_mul(dist_vec, last_vec)

        #return self.hd_mul(sensor_vec, last_vec)
#        if train:
#            return self.hd_threshold(last_vec + self.sensor_weight*sensor_vec + dist_vec)
#        else:
#            return self.hd_threshold(last_vec + sensor_vec + dist_vec)
        #random_vec = np.squeeze(self.create_bipolar_mem(1,self.dim))


        return self.hd_threshold(last_vec + sensor_vec + dist_vec)


    #methods 2 and 3
    def encode_sensors_directional(self, sensor_in, train, method):
        # Encode sensory data into HD space
        # Currently binds together all sensor inputs
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - sensor_vec: bipolar HD vector
        sensor_vec = np.empty((self.dim,4), dtype = np.int8) #bind
        for i,sensor_val in enumerate(sensor_in[:4]):
            permuted_vec = self.hd_sensor_vals[sensor_val,:]
            for j in range(i):
                # permute hd_sensor_val based on the corresponding sensor id
                permuted_vec = self.hd_perm(permuted_vec)

            binded_sensor = self.hd_mul(self.hd_sensor_ids[i,:],permuted_vec)
            sensor_vec[:,i] = binded_sensor

        xsensors = self.hd_mul(sensor_vec[:,0],sensor_vec[:,1])
        ysensors = self.hd_mul(sensor_vec[:,2],sensor_vec[:,3])


        if sensor_in[4] > 0:
            xval = self.hd_sensor_dist[2]
        elif sensor_in[4] < 0:
            xval = self.hd_sensor_dist[0]
        else:
            xval = self.hd_sensor_dist[1]

        if sensor_in[5] > 0:
            yval = self.hd_sensor_dist[2]
        elif sensor_in[5] < 0:
            yval = self.hd_sensor_dist[0]
        else:
            yval = self.hd_sensor_dist[1]
        yval = self.hd_perm(yval)

        xdist_vec = self.hd_mul(self.hd_sensor_ids[4,:], xval)
        ydist_vec = self.hd_mul(self.hd_sensor_ids[5,:], yval)
        dist_vec = self.hd_mul(xdist_vec, ydist_vec)

        xsense = self.hd_mul(xsensors,xdist_vec)
        ysense = self.hd_mul(ysensors,ydist_vec)

        last_vec = self.hd_sensor_last[sensor_in[6],:]
        #last_vec = self.hd_mul(dist_vec, last_vec)

        #return self.hd_mul(sensor_vec, last_vec)
#        if train:
#            return self.hd_threshold(last_vec + self.sensor_weight*sensor_vec + dist_vec)
#        else:
#            return self.hd_threshold(last_vec + sensor_vec + dist_vec)
        #random_vec = np.squeeze(self.create_bipolar_mem(1,self.dim))

        if method==2:
            ## method 2
            return self.hd_threshold(xsense + ysense + last_vec)
        elif method==3:
            ## method 3
            return self.hd_mul(self.hd_mul(xsensors,ysensors), self.hd_threshold(xdist_vec + ydist_vec + last_vec))




    def new_condition(self, condition_vec, threshold):
        dist = np.matmul(condition_vec, self.hd_threshold(self.hd_cond_vec), dtype = np.int)
        pct = dist/self.dim
        if (pct > threshold):
            return 0
        else:
            return 1

    def train_sample(self, sensor_in, act_in):
        # Multiply encoded sensor vector with actuator vector
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        #   - act_in: integer representing actuator action
        # outputs:
        #   - sample_vec: bipolar HD vector

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        sensor_vec = self.encoding_method(sensor_in,True)
        if self.new_condition(sensor_vec, self.threshold_known):
            act_vec = self.hd_actuator_vals[act_in,:]
            sample_vec = self.hd_mul(sensor_vec,act_vec)
            self.hd_cond_vec += sensor_vec
            self.num_cond += 1
        else:
            sample_vec = np.zeros((self.dim), dtype=np.int8)
            self.num_thrown += 1

        return sample_vec

    def test_sample(self, sensor_in):
        # Determine actuator action given sensory data
        # inputs:
        #   - sensor_in: array of binary flags (length 4)
        # outputs:
        #   - act_out: integer representing decided actuator action

        #for method 1, use encode_sensors
        #for method 2 and 3, use encoder_sensors_directional and change the last line of that function depending on the method
        sensor_vec = self.encoding_method(sensor_in,False)
        #unbind_vec = self.hd_mul(sensor_vec,self.hd_threshold(self.hd_program_vec))
        unbind_vec = self.hd_mul(sensor_vec,self.hd_program_vec)
        #act_out = self.search_actuator_vals(unbind_vec)
        act_out = self.softmax_actuator_vals(unbind_vec, self.softmax_param)

        return act_out

    def train_from_file(self, file_in):
        # Build the program HV from a text file of recorded moves
        # inputs:
        #   -file_in: filename for the recorded moves
        # Currently, the hd_program_vec is not being thresholded
        game_data = np.loadtxt(file_in, dtype = np.int8, delimiter=',')
        sensor_vals = game_data[:,:-1]
        actuator_vals = game_data[:,-1]
        n_samples = game_data.shape[0]
#        program_vec_b4thresh = np.zeros((self.dim,),dtype=np.int8)
#        for sample in range(n_samples):
#            sample_vec = self.train_sample(sensor_vals[sample,:],actuator_vals[sample])
#            program_vec_b4thresh = program_vec_b4thresh + sample_vec
#        if n_samples%2 == 0:
#            random_vec = np.squeeze(self.create_bipolar_mem(1,self.dim))
#            program_vec_b4thresh = program_vec_b4thresh + random_vec
#        self.hd_program_vec = self.hd_threshold(program_vec_b4thresh)
        for sample in range(n_samples):
            sample_vec = self.train_sample(sensor_vals[sample,:],actuator_vals[sample])
            if any(sample_vec):
                self.output_vectors.append(sample_vec)
                self.output_actuators.append(actuator_vals[sample])
            self.hd_program_vec = self.hd_program_vec + sample_vec

        pickle_vectors = open('output_vectors.pckl', 'wb')
        pickle_actuators = open('output_actuators.pckl', 'wb')
        pickle.dump(self.output_vectors, pickle_vectors)
        pickle.dump(self.output_actuators, pickle_actuators)
        pickle_vectors.close()
        pickle_actuators.close()

        return

    def test_from_file(self, file_in):
        # Prints out input sensor data and resulting output
        # inputs:
        #   -file_in: filename for the recorded moves
        # Currently, the hd_program_vec is not being thresholded
        game_data = np.loadtxt(file_in, dtype = np.int, delimiter=',')
        sensor_vals = game_data[:,:-1]
        actuator_vals = game_data[:,-1]
        n_samples = game_data.shape[0]
        correct = 0
        valid = 0

        for sample in range(n_samples):
            #print("sensor inputs: {}".format(game_data[sample]))
            act_out = self.test_sample(sensor_vals[sample,:])
            #print("guessed output: {} \t correct output: {}".format(act_out, actuator_vals[sample]))
            if (act_out == actuator_vals[sample]):
                correct += 1
            if self.is_valid_move(sensor_vals[sample,:],act_out):
                valid += 1
        print("Accuracy: {}".format(correct/n_samples))
        print("Valid: {}".format(valid/n_samples))
        return

    def is_valid_move(self, sensor_in, move_in):
        if sensor_in[move_in]==1:
            valid = False
        else:
            valid = True

        return valid







