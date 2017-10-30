# -*- coding: utf-8 -*-

import cherrypy
import argparse
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket
from cnn_dqn_agentRobo import CnnDqnAgent
import msgpack
import io
from PIL import Image
from PIL import ImageOps
import threading
import numpy as np

import matplotlib.pyplot as plt
#Q関数のplot
def pause_Q_plot(q):
    actions = [0,1,2]

    plt.cla()

    plt.xticks([0, 1, 2])
    plt.xlabel("Action") # x軸のラベル
    plt.ylabel("Q_Value") # y軸のラベル
    plt.ylim(-1.1, 1.1)  # yを-1.1-1.1の範囲に限定
    plt.xlim(-0.5, 2.5) # xを-0.5-2.5の範囲に限定
    plt.hlines(y=0, xmin=-0.5, xmax= 2.5, colors='r', linewidths=2) #y=0の直線

    max_q_abs = max(abs(q))
    if max_q_abs > 0:
        q = q / float(max_q_abs)

    plt.bar(actions,q,align="center")

    plt.pause(1.0 / 10**10) #引数はsleep時間


parser = argparse.ArgumentParser(description='ml-agent-for-unity')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help=u'websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help=u'server ip')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help=u'GPU ID (negative value indicates CPU)')


parser.add_argument('--draw', '-d', action = "store_true",
                    help=u'Draw Bar of Q-value flags')
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help=u'reward log file name')
parser.add_argument('--test', '-t', action = "store_true",
                    help=u'TEST frags, False => Train')
parser.add_argument('--succeed', '-s', action = "store_true",
                    help=u'Modelを引き継いでトレーニングをするか')
parser.add_argument('--model_num', '-m', default=0,type=int,
                    help=u'最初にロードするモデルの番号')
parser.add_argument('--velocity', '-v', default=0,type=int,
                    help=u'Agentの速度')
parser.add_argument('--episode', '-e', default=1, type=int,
                    help=u'logファイルに書き込む際のエピソードの数')



args = parser.parse_args()

'''
# Qの値を描画するキャンバス作成
if args.draw:
    #q = np.array([0,0,0])
    #pause_Q_plot(q)

    for i in range(50):
        print i
        q_now = np.random.rand(3)
        #pause_Q_plot(q_now)
'''



class Root(object):
    @cherrypy.expose
    def index(self):
        return 'some HTML with a websocket javascript connection'

    @cherrypy.expose
    def ws(self):
        # you can access the class instance through
        handler = cherrypy.request.ws_handler


class AgentServer(WebSocket):

    agent = CnnDqnAgent()#cnn_dqn_agent.pyの中のCnnDqnAgentクラスのインスタンス
    agent_initialized = False

    thread_event = threading.Event()#threading -> Eventの中にWait,Setがある
    #reward_sum = 0

    #depthImageをベクトルreshape,agent_initの引数に使用
    depth_image_dim = 32 * 32
    depth_image_count = 1

    log_file = args.log_file
    gpu = args.gpu
    draw = args.draw
    test = args.test
    succeed = args.succeed
    model_num = args.model_num
    velocity = args.velocity
    episode_num = args.episode #行ったエピソードの数
    cycle_counter = model_num #agentの行動回数、logファイルのX軸の値

    print u"------------------------------------------------"
    print u"Velocity = %d"%(velocity)
    print u"Unity側のAgentのMaxSpeedがあってるか確認"
    print u"./Model%dディレクトリが存在するか確認"%(velocity)
    print u"logファイルがあってるか確認"
    print u"------------------------------------------------"


    def send_action(self, action):
        dat = msgpack.packb({"command": str(action)})
        self.send(dat, binary=True)

    def received_message(self, m):
        payload = m.data
        dat = msgpack.unpackb(payload)

        image = []
        for i in xrange(self.depth_image_count):
            image.append(Image.open(io.BytesIO(bytearray(dat['image'][i]))))
        depth = []
        for i in xrange(self.depth_image_count):
            d = (Image.open(io.BytesIO(bytearray(dat['depth'][i]))))
            #depth画像は一次元ベクトルにreshape
            depth.append(np.array(ImageOps.grayscale(d)).reshape(self.depth_image_dim))

        observation = {"image": image, "depth": depth}
        reward = dat['reward']
        end_episode = dat['endEpisode']

        lastZ = dat['score']

        if not self.agent_initialized:
            self.agent_initialized = True
            print ("initializing agent...")
            #depth_image_dimが引数で使われるのはここだけ
            self.agent.agent_init(
                use_gpu=self.gpu,
                depth_image_dim=self.depth_image_dim * self.depth_image_count,
                test= self.test,
                succeed = self.succeed,
                model_num = self.model_num,
                velocity = self.velocity)

            action = self.agent.agent_start(observation,self.episode_num)
            self.send_action(action)

            #logファイルへの書き込み
            if not self.succeed:
                with open(self.log_file, 'w') as the_file:
                    the_file.write('Cycle,Score,Episode \n')

        else:
            self.thread_event.wait()
            self.cycle_counter += 1
            #self.reward_sum += reward

            if end_episode:
                self.agent.agent_end(reward,lastZ)

                #logファイルへの書き込み
                with open(self.log_file, 'a') as the_file:
                    the_file.write(str(self.cycle_counter) +
                               ',' + str(lastZ) +
                               ',' + str(self.episode_num) + '\n')
                #self.reward_sum = 0


                if(args.test and self.episode_num % 50 == 0):
                    #self.episode_num = 0
                    self.cycle_counter = 0
                    self.model_num += 10000
                    self.agent.q_net.load_model(self.model_num,self.velocity)

                self.episode_num += 1
                action = self.agent.agent_start(observation,self.episode_num)  # TODO
                self.send_action(action)

            else:
                action, eps, q_now, obs_array = self.agent.agent_step(reward, observation)

                # draw Q value
                if args.draw:
                    pause_Q_plot(q_now.ravel())


                self.send_action(action)
                self.agent.agent_step_update(reward, action, eps, q_now, obs_array)

        self.thread_event.set()



cherrypy.config.update({'server.socket_host': args.ip,
                        'server.socket_port': args.port})
WebSocketPlugin(cherrypy.engine).subscribe()
cherrypy.tools.websocket = WebSocketTool()
cherrypy.config.update({'engine.autoreload.on': False})
config = {'/ws': {'tools.websocket.on': True,
                  'tools.websocket.handler_cls': AgentServer}}
cherrypy.quickstart(Root(), '/', config)
