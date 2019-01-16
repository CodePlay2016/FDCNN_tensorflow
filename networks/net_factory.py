#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:37:47 2018

@author: codeplay2018
"""
import networks.vgg as vgg
import networks.calex as calex
import networks.simpnets as simpnets
import networks.speednets as speednets
import networks.shallow_cnn as shallownets
import networks.vgg_for_visual as vgg4vis
import networks.resnet as resnet
import networks.cardinet as cardinet

networks_map = {'cvgg19': vgg.cvgg19,
                'cvgg19_2': vgg.cvgg19_2,
                'cvgg19_3': vgg.cvgg19_3,
                'calex': calex.cAlex,

                'cvgg19v': vgg4vis.cvgg19_with_pindex,
                'dvgg19v': vgg4vis.de_cvgg19,

                'sphere20': resnet.sphere_net20,
                'cardinet': cardinet.cardinet,
                
                'simpnet1': simpnets.simpnet1,
                'simpnet2': simpnets.simpnet2,
                
                'shallow_cnn': shallownets.shallow1,
                'shallow_cnn_0': shallownets.shallow2,
                'speednet1': speednets.speednet1,
                'speednet2': speednets.speednet2
                }

def get_network(name):
    return networks_map[name]
    