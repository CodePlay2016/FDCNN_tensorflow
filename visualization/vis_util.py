#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 20:10:29 2018

@author: codeplay2018
"""

import matplotlib as mpl

def set_tick_font(axe, font_name='Time New Roman'):
    for tick in axe.get_xticklabels():
        tick.set_fontname(font_name)
    for tick in axe.get_yticklabels():
        tick.set_fontname(font_name)
