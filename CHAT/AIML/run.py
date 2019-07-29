#!/usr/bin/python
# coding:utf8
"""
@author: Cong Yu
@time: 2019-07-27 15:45
"""
import aiml.Kernel as K

alice = K()
alice.learn("cn-test.aiml")
while True:
    print(alice.respond(input('Alice请您提问...>>')))
