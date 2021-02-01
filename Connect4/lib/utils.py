#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:46:34 2021

@author: Ayman Jabri

Contents:
    1- Buffer class: stores connect4 experiences keeping history. The class
                    augments empty states to the first states
    2- unpack batch function: upack a batch from buffer into trainable 
"""
import torch
import random
import numpy as np
from typing import List
from collections import deque, namedtuple
from lib.cylib import functions_cy as cy


Experience = namedtuple('Experience', ['state', 'action', 'player'])

########################
#### Replay Buffer #####
########################


class ReplayBuffer(object):
    def __init__(self, init_state=1797558, maxlen=200_000, history=4):
        self.empty_state = init_state
        self.maxlen = maxlen
        self.history = history
        self.buffer = deque(maxlen=maxlen)
        self.buffer_backup = deque(maxlen=maxlen)
        self.empty = Experience(init_state, -1, -1)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def store(self, episode: List):
        """
        Store one played episode adding `h` history moves to each experience
        (automatically adds empty h-i states at the begining) 

        Parameters
        ----------
        episode : List
            A full episode consists of:
                - Moves as named tuples: Experience(state, action, player)
                - Value as Int: represents if player 0 won, added to the end 
        """
        assert isinstance(episode, List)
        value = episode.pop()
        exp = [self.empty] * (self.history-1) + episode
        lenght = len(exp)
        for idx in range(lenght-self.history+1):
            self.buffer.append((exp[idx:idx+4],value))

    # def _store_backup(self, episode: List):
    #     """
    #     Store a played episode accounting for history but WITHOUT
    #     adding empty states to the first few board states

    #     Parameters
    #     ----------
    #     episode : List
    #         A full episode consists of Experiences that include: state, action, player
    #     """
    #     assert isinstance(episode, List)
    #     lenght = len(episode)
    #     a = 0
    #     for idx in range(1, lenght+1):
    #         self.buffer_backup.append(episode[a:idx])
    #         if idx >= self.history:
    #             a += 1

    def sample(self, batch_size):
        batch_size = min(self.__len__(), batch_size)
        return random.sample(self.buffer, batch_size)


def unpack_batch(batch, device='cpu'):
    states = []
    actions = []
    values = []
    players = []
    exp,values = zip(*batch)
    for moves in exp:
        state, action, player = zip(*moves)
        st_res = []
        for s in state:
            st_res.extend(cy.binary_to_array(s))
        states.append(np.array(st_res, copy=False))
        actions.append(action[-1])
        players.append(player[-1])
    states_v = torch.Tensor(np.array(states,copy=False)).to(device)
    return states_v, actions, players, values
