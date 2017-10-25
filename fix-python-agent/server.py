# -*- coding: utf-8 -*-

import cherrypy
import argparse
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket
from cnn_dqn_agent import CnnDqnAgent
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


print "----------------------------------"
print "This is Okuyama's python-agent !!!"
print "----------------------------------"

parser = argparse.ArgumentParser(description='ml-agent-for-unity')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help=u'websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help=u'server ip')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help=u'GPU ID (negative value indicates CPU)')
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help=u'reward log file name')

parser.add_argument('--test', '-t', action = "store_true",
                    help=u'TEST flags, False => Train')
parser.add_argument('--draw', '-d', action = "store_true",
                    help=u'Draw bar of Q-value flags')


parser.add_argument('--succeed', '-s', default=0, type=int,
                    help=u'cycle_counterの値, cnn_dqn_agentのStep数やepsilon,ModelNameがこの値で決まる')

parser.add_argument('--episode', '-e', default=1, type=int,
                    help=u'logファイルに書き込む際のエピソードの数,cnn_dqn_agentとは関係なし')

parser.add_argument('--model', '-m', default='Model/best_model',
                    help=u'name of load model(default : best_model)')

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

    cycle_counter = args.succeed#agentの行動回数、logファイルのX軸の値
    episode_num = args.episode #行ったエピソードの数

    thread_event = threading.Event()#threading -> Eventの中にWait,Setがある
    log_file = args.log_file
    model_name = args.model
    #reward_sum = 0

    #depthImageをベクトルreshape,agent_initの引数に使用
    depth_image_dim = 32 * 32
    depth_image_count = 1

    model_num = 10000


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

        lastZ = dat['lastZ']

        if not self.agent_initialized:
            self.agent_initialized = True
            print ("initializing agent...")
            #depth_image_dimが引数で使われるのはここだけ
            self.agent.agent_init(
                use_gpu=args.gpu,
                depth_image_dim=self.depth_image_dim * self.depth_image_count,
                test= args.test,
                model_name = self.model_name,
                succeed_num = args.succeed)

            action = self.agent.agent_start(observation,self.episode_num)
            self.send_action(action)

            #logファイルへの書き込み
            #if args.test is False and args.succeed<=0:
            if args.succeed<=0:
                with open(self.log_file, 'w') as the_file:
                    the_file.write('Cycle,Score,Episode \n')

        else:
            self.thread_event.wait()
            self.cycle_counter += 1
            #self.reward_sum += reward

            if end_episode:
                self.agent.agent_end(reward,lastZ)

                #logファイルへの書き込み
                #if args.test is False:
                with open(self.log_file, 'a') as the_file:
                    the_file.write(str(self.cycle_counter) +
                               ',' + str(lastZ) +
                               ',' + str(self.episode_num) + '\n')
                #self.reward_sum = 0


                if(args.test and self.episode_num % 20 == 0):
                    #self.episode_num = 0
                    self.cycle_counter = 0

                    self.model_num += 10000
                    self.model_name = "%dcycle_model_hoge"%(self.model_num)
                    #print "ok ok ok ok"

                    self.agent.model_load(args.test, args.succeed, self.model_name)
                    #self.log_file = "reward%d.log"%(self.model_num)
                    #with open(self.log_file, 'w') as the_file:
                        #the_file.write('Cycle,Score,Episode \n')
                    #print "ok ok ok ok ok ok"


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
