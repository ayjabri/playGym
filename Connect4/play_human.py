# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:39:53 2021

@author: ayman
"""
import joblib
import torch
import argparse
import random
import numpy as np
from lib import game, mcts, model


def get_action(game, net, mcts_, cur_state,cur_player, search_depth, batch_size, device):
    if net is None:
        action = int(input('Enter column number to place a stone in (0->6): '))
        return action
    mcts_.search_batch(search_depth, batch_size, cur_state, cur_player, net, device)
    prob, _ = mcts_.get_policy_value(cur_state)
    return int(np.argmax(prob))
    


def play(net1,net2=None,mcts_store=None, batch_size=16,
         search_depth=10,steps_before_tau=10,device='cpu'):   
    if net2 is None:
        print('*'*10,' Human vs Computer', '*'*10)
        print('\t    You are the Black player  \n')
    nets = [net1, net2]
    connect = game.Connect()
    if mcts_store is None:
        mcts_store = [mcts.MCTS(connect), mcts.MCTS(connect)]
    cur_player = random.choice(range(2))
    cur_state = connect.init_state
    player = 'White' if cur_player ==0 else 'Black'
    print(f'########## {player} plays first ##########\n\n')
    while True:
        connect.print_board(cur_state)
        action = get_action(connect,
                            nets[cur_player],
                            mcts_store[cur_player],
                            cur_state,
                            cur_player,
                            search_depth,
                            batch_size,
                            device=device
                            )
        cur_state, won = connect.move(cur_state, action, cur_player)
        if won:
            connect.print_board(cur_state)
            print(f'Player {cur_player} wins')
            break
        cur_player = 1-cur_player
    return mcts_store
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_120_62500.dat', help='trained model')
    parser.add_argument('--depth', type=int, default=10, help='MCTS search depth')
    parser.add_argument('--batch', type=int, default=16, help='MCTS batch size')
    parser.add_argument('--mcts', default=None, help='trained mcts')
    parser.add_argument('--cuda', action='store_true', default=False, help='Activate GPU')
    args = parser.parse_args()
    
    device = 'cuda' if args.cuda else 'cpu'
    if args.mcts is not None:
        mcts_store = joblib.load(args.mcts)
    else:
        mcts_store = None
    net = model.Net(model.OBS_SHAPE, 7).to(device).eval()
    net.load_state_dict(torch.load(args.model, map_location=device))
    mcts_store = play(net, mcts_store=mcts_store, batch_size=args.batch, search_depth=args.depth, device=device)
    if args.mcts is not None:
        joblib.dump(mcts_store, args.mcts)
        
        
    
# =======
# # -*- coding: utf-8 -*-
# """
# Created on Fri Jan 29 11:39:53 2021

# @author: ayman
# """
# import argparse
# import random
# import numpy as np
# from lib import mcts, model
# from typing import Optional, List, Tuple
# from lib.game import game


# def get_action(game, net, mcts_, cur_state,cur_player, search_depth, batch_size):
#     if net is None:
#         action = int(input('Enter column number to place a stone in (0->6): '))
#         return action
#     mcts_.search_batch(search_depth, batch_size, cur_state, cur_player, net)
#     prob, _ = mcts_.get_policy_value(cur_state)
#     return int(np.argmax(prob))



# def play(net1,net2=None,mcts_store=None, batch_size=16,
#          search_depth=10,steps_before_tau=10,device='cpu'):
#     if net2 is None:
#         print('*'*10,' Human vs Computer', '*'*10)
#         print('\t    You are the Black player  \n')
#     nets = [net1, net2]
#     connect = game.Connect()
#     if mcts_store is None:
#         mcts_store = [mcts.MCTS(connect), mcts.MCTS(connect)]
#     cur_player = random.choice(range(2))
#     cur_state = connect.init_state
#     player = 'White' if cur_player ==0 else 'Black'
#     print(f'########## {player} plays first ##########\n\n')
#     while True:
#         connect.print_board(cur_state)
#         action = get_action(connect,
#                             nets[cur_player],
#                             mcts_store[cur_player],
#                             cur_state,
#                             cur_player,
#                             search_depth,
#                             batch_size
#                             )
#         cur_state, won = connect.move(cur_state, action, cur_player)
#         if won:
#             connect.print_board(cur_state)
#             print(f'Player {cur_player} wins')
#             break
#         cur_player = 1-cur_player


# if __name__=='__main__':
#     parser = argparse.PARSER()
#     parser.add_argument('-net1', default=None, help='path to net1 dict_state')
#     args = parser.parse_args()
#     net = model.Net()
#     play(net)
# >>>>>>> 407adb5018e7855d1d4d69c45901cda70c32f9aa
