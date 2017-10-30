# -*- coding: utf-8 -*-

import six.moves.cPickle as pickle
import copy
import os
import numpy as np

import matplotlib.pyplot as plt

from chainer import cuda

from cnn_feature_extractorRobo import CnnFeatureExtractor
from q_netRobo import QNet

class CnnDqnAgent(object):
    policy_frozen = False
    #deltaの減少量
    epsilon_delta = 1.0 / 10 ** 4.4
    #deltaの最小値
    min_eps = 0.1
    #アクションリスト(数字じゃなくても大丈夫)
    actions = [0, 1, 2]

    cnn_feature_extractor = 'alexnet_feature_extractor.pickle' #1
    model = 'bvlc_alexnet.caffemodel' #2
    model_type = 'alexnet' #3

    image_feature_dim = 256 * 6 * 6
    image_feature_count = 1


    def _observation_to_featurevec(self, observation):
        # TODO clean
        if self.image_feature_count == 1:
            return np.r_[self.feature_extractor.feature(observation["image"][0]),
                         observation["depth"][0]]
        elif self.image_feature_count == 4:
            return np.r_[self.feature_extractor.feature(observation["image"][0]),
                         self.feature_extractor.feature(observation["image"][1]),
                         self.feature_extractor.feature(observation["image"][2]),
                         self.feature_extractor.feature(observation["image"][3]),
                         observation["depth"][0],
                         observation["depth"][1],
                         observation["depth"][2],
                         observation["depth"][3]]
        else:
            print("not supported: number of camera")

    def agent_init(self, **options):
        self.use_gpu = options['use_gpu']
        self.depth_image_dim = options['depth_image_dim']

        self.q_net_input_dim = self.image_feature_dim * self.image_feature_count + self.depth_image_dim

        if os.path.exists(self.cnn_feature_extractor):
            print("loading... " + self.cnn_feature_extractor),
            self.feature_extractor = pickle.load(open(self.cnn_feature_extractor))
            print("done")

        else:
            self.feature_extractor = CnnFeatureExtractor(self.use_gpu, self.model, self.model_type, self.image_feature_dim)
            pickle.dump(self.feature_extractor, open(self.cnn_feature_extractor, 'w'))
            print("pickle.dump finished")

        self.q_net = QNet(self.use_gpu, self.actions, self.q_net_input_dim)

        test = options['test']
        succeed = options['succeed']
        model_num = options['model_num']
        self.velocity = options["velocity"] #save_modelでもしようするため,selfをつけた

        self.policy_frozen = test

        self.time = model_num
        non_exploration = max(self.time - self.q_net.initial_exploration , 0)
        self.epsilon = max(1.0 - non_exploration * self.epsilon_delta , self.min_eps)
        print "epsilon = ",self.epsilon

        if(test or succeed):
            self.q_net.load_model(model_num,self.velocity)


    # 行動取得系,state更新系メソッド
    def agent_start(self, observation, episode_num):
        print "----------------------------------"
        print "Episode %d Start"%(episode_num)
        print "----------------------------------"
        obs_array = self._observation_to_featurevec(observation)
        # Initialize State
        self.state = np.zeros((self.q_net.hist_size, self.q_net_input_dim), dtype=np.uint8)
        self.state[0] = obs_array
        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Generate an Action e-greedy
        action, q_now = self.q_net.e_greedy(state_, self.epsilon)
        return_action = action

        # Update for next step
        self.last_action = copy.deepcopy(return_action)
        self.last_state = self.state.copy()

        # 更新するだけで使ってない
        self.last_observation = obs_array

        return return_action

    # 行動取得系,state更新系メソッド
    def agent_step(self, reward, observation):
        # rewardは使ってない

        obs_array = self._observation_to_featurevec(observation)

        #obs_processed = np.maximum(obs_array, self.last_observation)  # Take maximum from two frames

        # Compose State : 4-step sequential observation
        if self.q_net.hist_size == 4:
            self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_array], dtype=np.uint8)
        elif self.q_net.hist_size == 2:
            self.state = np.asanyarray([self.state[1], obs_array], dtype=np.uint8)
        elif self.q_net.hist_size == 1:
            self.state = np.asanyarray([obs_array], dtype=np.uint8)
        else:
            print("self.DQN.hist_size err")

        # q_funcに入れる際は(サンプル数,hist_size,q_net_input_dim)
        # 例:np.ndarray(shape=(self.replay_size, self.hist_size, self.dim))
        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.q_net_input_dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Exploration decays along the time sequence
        if self.policy_frozen is False:  # Learning ON/OFF
            if self.q_net.initial_exploration < self.time: #timeが1000を超えたら
                self.epsilon -= self.epsilon_delta
                if self.epsilon < self.min_eps:
                    self.epsilon = self.min_eps
                eps = self.epsilon

            #最初に1000回ランダムに行動
            else:  # Initial Exploation Phase
                print("Initial Exploration : %d/%d steps" % (self.time, self.q_net.initial_exploration)),
                eps = 1.0
        else:  # Evaluation
            print("Policy is Frozen")
            eps = 0

        # Generate an Action by e-greedy action selection
        action, q_now = self.q_net.e_greedy(state_, eps)

        return action, eps, q_now, obs_array

   # 学習系メソッド
    def agent_step_update(self, reward, action, eps, q_now, obs_array):
        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.last_action, reward, self.state, False)
            self.q_net.experience_replay(self.time)

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()

        # Simple text based visualization
        if self.use_gpu >= 0:
            q_max = np.max(q_now.get())
        else:
            q_max = np.max(q_now)

        print('Step:%d  Action:%d  Reward:%.1f  Epsilon:%.6f  Q_max:%3f' % (
            self.time, self.q_net.action_to_index(action), reward, eps, q_max))

        # Updates for next step , 更新するだけで使ってない
        self.last_observation = obs_array

        if self.policy_frozen is False:
            self.last_action = copy.deepcopy(action)
            self.last_state = self.state.copy()

            # save model
            if self.q_net.initial_exploration < self.time and np.mod(self.time,self.q_net.save_model_freq) == 0:
                print "------------------Save Model------------------"
                self.q_net.save_model(self.time,self.velocity)

        # Time count
        self.time += 1

    # 学習系メソッド
    def agent_end(self, reward, lastZ):  # Episode Terminated
        print('episode finished. Reward:%.1f / Epsilon:%.6f' % (reward, self.epsilon))

        print "Episode time is %d"%(lastZ)

        # Learning Phase
        if self.policy_frozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.last_action, reward, self.last_state,True)
            self.q_net.experience_replay(self.time)

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print("Model Updated")
            self.q_net.target_model_update()


        if self.policy_frozen is False:
            # Model Save
            if self.q_net.initial_exploration < self.time and np.mod(self.time,self.q_net.save_model_freq) == 0:
                print "------------------Save Model------------------"
                self.q_net.save_model(self.time)

        # Time count
        self.time += 1
