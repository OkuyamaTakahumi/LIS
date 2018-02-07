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

parser = argparse.ArgumentParser(description='ml-agent-for-unity')
parser.add_argument('--log-file', '-l', default='reward.log', type=str,
                    help=u'reward log file name')
parser.add_argument('--folder', '-f', default='Model0', type=str,
                    help=u'モデルの存在するフォルダ名')
parser.add_argument('--model_num', '-m', default=0,type=int,
                    help=u'最初にロードするモデルの番号')
parser.add_argument('--test', '-t', action = "store_true",
                    help=u'TEST frags, False => Train')
parser.add_argument('--episode', '-e', default=1, type=int,
                    help=u'logファイルに書き込む際のエピソードの数')
parser.add_argument('--draw', '-d', action = "store_true",
                    help=u'Draw Bar of Q-value flags')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help=u'GPU ID (negative value indicates CPU)')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help=u'websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help=u'server ip')

args = parser.parse_args()

class Root(object):
    @cherrypy.expose
    def index(self):
        return 'some HTML with a websocket javascript connection'

    @cherrypy.expose
    def ws(self):
        # you can access the class instance through
        handler = cherrypy.request.ws_handler


class AgentServer(WebSocket):
    test_num = 5000 # 1つのモデルをためすテストの回数

    agent = CnnDqnAgent()#cnn_dqn_agent.pyの中のCnnDqnAgentクラスのインスタンス
    agent_initialized = False

    thread_event = threading.Event()#threading -> Eventの中にWait,Setがある
    reward_sum = 0
    depth_image_dim = 32 * 32 #receive後のreshapeに使うのでここで定義
    image_count = 1 # 1cycleでUnityのAgentから送られてくる画像の枚数

    log_file = args.log_file
    gpu = args.gpu
    draw = args.draw
    test = args.test
    episode_num = args.episode #行ったエピソードの数
    folder = args.folder
    model_num = args.model_num

    cycle_counter = 0

    print u"------------------------------------------------"
    print u"./%sディレクトリが存在するか確認"%(folder)
    print u"logファイルがあってるか確認"
    print u"------------------------------------------------"


    def send_action(self, action):
        dat = msgpack.packb({"command": str(action)})
        self.send(dat, binary=True)

    def received_message(self, m):
        try:
            payload = m.data
            dat = msgpack.unpackb(payload)

            image = []
            for i in xrange(self.image_count):
                image.append(Image.open(io.BytesIO(bytearray(dat['image'][i]))))
            depth = []
            for i in xrange(self.image_count):
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
                    depth_image_dim=self.depth_image_dim,
                    image_count=self.image_count,
                    use_gpu=self.gpu,
                    test= self.test,
                    folder = self.folder,
                    model_num = self.model_num)

                action = self.agent.agent_start(observation)
                self.send_action(action)
                print "send"

                #logファイルへの書き込み
                if not self.succeed:
                    with open(self.log_file, 'w') as the_file:
                        the_file.write('Cycle,Score,Episode \n')

                if(args.draw):
                    self.fig, self.ax1 = plt.subplots(1, 1)

            else:
                self.thread_event.wait()
                self.cycle_counter += 1
                self.reward_sum += reward

                if end_episode:
                    self.agent.agent_end(reward)
                    #logファイルへの書き込み
                    with open(self.log_file, 'a') as the_file:
                        the_file.write(str(self.cycle_counter) +
                                   ',' + str(lastZ) +
                                   ',' + str(self.episode_num) + '\n')
                    print "Last Player's Z is %d"%(lastZ)
                    self.reward_sum = 0

                    if(args.test and self.episode_num % self.test_num == 0):
                        self.model_num += 10000
                        self.agent.q_net.load_model(self.model_num,self.velocity)

                    self.episode_num += 1

                    print "----------------------------------"
                    print "Episode %d Start"%(self.episode_num)
                    print "----------------------------------"

                    action = self.agent.agent_start(observation)  # TODO
                    self.send_action(action)

                else:
                    action, eps, q_now, obs_array = self.agent.agent_step(observation)
                    self.send_action(action)
                    self.agent.agent_step_update(reward, action, eps, q_now)

                    if args.draw:
                        pause_Q_plot(q_now.ravel())

            self.thread_event.set()
        except:
            import traceback
            import sys
            traceback.print_exc()
            sys.exit()

    def pause_Image_plot(self, img):
        plt.cla()
        plt.imshow(img)
        plt.pause(1.0 / 10**10) #引数はsleep時間

    #Q関数のplot
    def pause_Q_plot(self, q):
        self.ax1.cla()
        actions = range(3)
        q = q[:3]
        max_q_abs = max(abs(q))
        if max_q_abs != 0:
            q = q / float(max_q_abs)

        self.ax1.set_xticks(actions)
        self.ax1.set_xticklabels(['Left','Forward','Right'], rotation=0, fontsize='small')
        self.ax1.set_xlabel("Action") # x軸のラベル
        self.ax1.set_ylabel("Q_Value") # y軸のラベル
        self.ax1.set_ylim(-1.1, 1.1)  # yを-1.1-1.1の範囲に限定
        self.ax1.set_xlim(-1, 4)
        self.ax1.hlines(y=0, xmin=-1, xmax=4, colors='r', linewidths=2) #y=0の直線

        self.ax1.bar(actions,q,align="center")
        plt.pause(1.0 / 10**10) #引数はsleep時間



cherrypy.config.update({'server.socket_host': args.ip,
                        'server.socket_port': args.port})
WebSocketPlugin(cherrypy.engine).subscribe()
cherrypy.tools.websocket = WebSocketTool()
cherrypy.config.update({'engine.autoreload.on': False})
config = {'/ws': {'tools.websocket.on': True,
                  'tools.websocket.handler_cls': AgentServer}}
cherrypy.quickstart(Root(), '/', config)
