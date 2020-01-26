import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import replay_memory
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition' , ('state' , 'action' , 'state_next' , 'reward') )
BATCH_SIZE = 32
CAPACITY = 100
GAMMA = 0.99


class Brain:
    def __init__(self , num_states , num_actions):
        self.num_actions = num_actions
        self.memory = replay_memory.ReplayMemory ( CAPACITY )

        self.model = nn.Sequential ()
        self.model.add_module ( 'fc1' , nn.Linear ( num_states , 32 ) )
        self.model.add_module ( 'relu1' , nn.ReLU () )
        self.model.add_module ( 'fc2' , nn.Linear ( 32 , 32 ) )
        self.model.add_module ( 'relu2' , nn.ReLU () )
        self.model.add_module ( 'fc3' , nn.Linear ( 32 , num_actions ) )

        print ( self.model )
        self.optimizer = optim.Adam(self.model.parameters(),lr=0.0001)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return      # 何もしない

        """ミニバッチの作成"""
        transitions = self.memory.sample(BATCH_SIZE)
        # (state, action, state_next, reward)xBATCH_SIZE を
        # state xBATCH_SIZE, action xBATCH_SIZE, ...に変換
        batch = Transition(*zip(*transitions))
        # バラバラのstate(1x4)をひとかたまりのtensorに結合する(4xBATCH_SIZE)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.state_next if s is not None])

        """教師信号となるQ(s, a)を求める"""
        self.model.eval()  # ネットワークを推論モードに切り替える


        # まずmodelにstateを入れた時の出力「actionごとのQ値(Q(s,0),Q(s,1))」を計算し、そのうちのaction_batchに格納されたアクションに対応するQ値の方を選択して出力する
        state_action_values = self.model(state_batch).gather(1, action_batch)

        mask_true_or_false = tuple(map(lambda s: s is not None, batch.state_next))
            # ...batch.state_nextの要素が Noneなら False そうでなければTrueをtupleにして返す

        non_final_mask = torch.tensor(mask_true_or_false)  # ...torch.ByteTensor(tuple)はTrue,Falseを1,0に変換する
        next_state_values = torch.zeros(BATCH_SIZE)

        """ next_stateが存在するstateについて、 next_state_values maxQ(s_n, a)を計算する。"""
        tmp = self.model(non_final_next_states)     # Q(s_n ,a)を計算
        tmp2 = tmp.max(1)            # Q(s_n,0)とQ(S_n,1)の大きい方を選択
        tmp3 = tmp2[0]
        tmp4 = tmp3.detach()
        next_state_values[non_final_mask] = tmp4  # next_stateがないstateのQ(s,a)には0が埋まる
        # next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + GAMMA * next_state_values

        """結合パラメータの更新"""
        self.model.train()  # ネットワークを訓練モードに切り替える
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()    # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action





if __name__ == '__main__':
    Brain ( 4 , 2 )
