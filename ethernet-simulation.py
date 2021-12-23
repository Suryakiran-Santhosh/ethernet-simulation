"""
University of California Davis Spring 2021
ECS 152A: Computer Networks
Project Ethernet Simulation
By: Suryakiran Santhosh
"""


import sys
import random
import simpy
import math
import numpy as np
import matplotlib.pyplot as plt


# First define some global variables. You should change values
class G:
    RANDOM_SEED = 33
    SIM_TIME = 100000  # This should be large
    SLOT_TIME = 1
    N = 10
    ARRIVAL_RATES = [0.001, 0.002, 0.003, 0.006, 0.012, 0.024]  # Check the submission guidelines
    RETRANMISSION_POLICIES = ["pp", "op", "beb", "lb"]
    LAMBDA = [0.03, 0.06, 0.12, 0.24, 0.48]
    LONG_SLEEP_TIMER = 1000000000


class Server_Process(object):
    def __init__(self, env, dictionary_of_nodes, retran_policy, slot_stat):
        self.env = env
        self.dictionary_of_nodes = dictionary_of_nodes
        self.retran_policy = retran_policy
        self.slot_stat = slot_stat

        self.current_slot = 1

        self.action = env.process(self.run())

        self.successes = 0
        self.collisions = 0
        self.idle = 0

    def run(self):
        print("Server process started")

        while True:
            # sleep for slot time
            yield self.env.timeout(G.SLOT_TIME)

            # Code to determine what happens to a slot and then update node variables accordingly based

            # METHOD as per TA and prof
            # increment slot time
            # loop thru dictionary and using the retran policy i would append to the q
            # from the remaining pkts based off of the slot time send (1 pkt per slot)
            # retranmission policy functions --> increments the slot time of node
            # stats: slot time and probability and slot time

            active_nodes = []  # an array that is used to map the values in the dictionary to the current slot or transmission queue

            # find all the active nodes

            for i in range(1, len(self.dictionary_of_nodes), 1):
                if(self.dictionary_of_nodes[i].next_slot_number == self.current_slot):
                    active_nodes.append(i)  # i is the key value of the node or the node number

            # retransmission ideologies
            if(self.retran_policy == "pp"):
                if(len(active_nodes) > 1):
                    # multiple nodes try to transmit in a slot which causes a collision
                    for i in active_nodes:
                        pp = self.current_slot + np.random.geometric(0.5)
                        self.dictionary_of_nodes[i].next_slot_number = pp
                    self.collisions += 1
                elif(len(active_nodes) == 1):
                    # successful transmission because only one active node in this slot
                    for i in active_nodes:
                        self.dictionary_of_nodes[i].next_slot_number = self.current_slot + 1
                        self.dictionary_of_nodes[i].packet_number -= 1
                    self.successes += 1
                else:
                    # the idle state case
                    self.idle += 1
                self.current_slot += 1
            elif(self.retran_policy == "op"):
                if(len(active_nodes) > 1):
                    # multiple nodes try to transmit in a slot which causes a collision
                    for i in active_nodes:
                        op = self.current_slot + np.random.geometric(1/G.N)
                        self.dictionary_of_nodes[i].next_slot_number = op
                    self.collisions += 1
                elif(len(active_nodes) == 1):
                    # successful transmission
                    for i in active_nodes:
                        self.dictionary_of_nodes[i].packet_number -= 1
                        self.dictionary_of_nodes[i].next_slot_number = self.current_slot + 1
                    self.successes += 1
                else:
                    self.idle += 1
                self.current_slot += 1
            elif(self.retran_policy == "lb"):
                # linear backoff
                # this algorithm will continously run until a success occurs
                # this while loop method was bought up during Ramanujan's office hours
                same_slot = True
                while(same_slot):
                    same_slot = False
                    if(len(active_nodes) > 1):
                        self.collisions += 1
                        for i in active_nodes:
                            x = np.random.randint(min(self.dictionary_of_nodes[i].retrans_attempts, 1024) + 1)
                            lb_next = self.current_slot + x
                            self.dictionary_of_nodes[i].next_slot_number = lb_next
                            self.dictionary_of_nodes[i].retrans_attempts += 1

                            # check to see if while loop should iterate again
                            if(self.dictionary_of_nodes[i].next_slot_number == self.current_slot):
                                same_slot = True
                    elif(len(active_nodes) == 1):
                        # the success case
                        self.successes += 1

                        # book keeping for the node object
                        for i in active_nodes:
                            self.dictionary_of_nodes[i].packet_number -= 1  # bc successful transmission
                            self.dictionary_of_nodes[i].next_slot_number = self.current_slot + 1  # progress in time
                            self.dictionary_of_nodes[i].retrans_attempts = 1  # to reinitialize a node
                    else:
                        self.idle += 1
            else:
                # binary exponential backoff
                same_slot = True
                while(same_slot):
                    same_slot = False

                    # collision case
                    if(len(active_nodes) > 1):
                        self.collisions += 1

                        for i in active_nodes:
                            K = min(self.dictionary_of_nodes[i].retrans_attempts, 10)
                            beb = self.current_slot + np.random.randint(2 ** K)
                            self.dictionary_of_nodes[i].next_slot_number = beb
                            self.dictionary_of_nodes[i].retrans_attempts += 1

                            # while loop iteration checker
                            if(self.dictionary_of_nodes[i].next_slot_number == self.current_slot):
                                same_slot = True
                    elif(len(active_nodes) == 1):
                        # success case
                        self.successes += 1

                        for i in active_nodes:
                            self.dictionary_of_nodes[i].packet_number -= 1
                            self.dictionary_of_nodes[i].next_slot_number = self.current_slot + 1
                            self.dictionary_of_nodes[i].retrans_attempts = 1
                    else:
                        # idle case
                        self.idle += 1

                    # increment time
                    self.current_slot += 1


class Node_Process:
    def __init__(self, env, id, arrival_rate):
        self.env = env
        self.id = id
        self.arrival_rate = arrival_rate

        # Other state variables
        self.packet_number = 0
        self.next_slot_number = 0
        self.retrans_attempts = 1

        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(random.expovariate(self.arrival_rate))

            print("Arrival Process Started: ", self.id)

            if self.packet_number == 0:
                self.next_slot_number = math.ceil(self.env.now)

            self.packet_number += 1


class Packet:
    def __init__(self, identifier, arrival_time, sl):
        self.identifier = identifier
        self.arrival_time = arrival_time
        self.slot_time = sl


class StatObject(object):
    def __init__(self):
        self.dataset = []

    def addNumber(self, x):
        self.dataset.append(x)


def main():
    print("Simulation Analysis of Random Access Protocols")
    random.seed(G.RANDOM_SEED)

    # command line argument definition
    if len(sys.argv) - 1 > 1:
        G.RETRANMISSION_POLICIES.clear()  # remove all the policies and add the policy that was inputted
        G.RETRANMISSION_POLICIES.append(sys.argv[2])

        G.N = int(sys.argv[1])  # command line number of nodes input

        G.ARRIVAL_RATES.clear()  # remove all
        G.ARRIVAL_RATES.append(float(sys.argv[len(sys.argv) - 1]))  # command line arrival rate input

    throughput_pp = []
    throughput_op = []
    throughput_lb = []
    throughput_beb = []
    Lambda = [0.03, 0.06, 0.12, 0.24, 0.48] * G.N

    for retran_policy in G.RETRANMISSION_POLICIES:
        for arrival_rate in G.ARRIVAL_RATES:
            env = simpy.Environment()
            slot_stat = StatObject()
            dictionary_of_nodes = {}  # I chose to pass the list of nodes as a
            # dictionary since I really like python dictionaries :)

            for i in list(range(1, G.N+1)):
                node = Node_Process(env, i, arrival_rate)
                dictionary_of_nodes[i] = node

            server_process = Server_Process(env, dictionary_of_nodes, retran_policy, slot_stat)
            env.run(until=G.SIM_TIME)

            # calculate throughput
            tp = (server_process.successes) / (server_process.current_slot)
            if(retran_policy == "pp"):
                throughput_pp.append(tp)
            elif(retran_policy == "op"):
                throughput_op.append(tp)
            elif(retran_policy == "lb"):
                throughput_lb.append(tp)
            else:  # beb case
                throughput_beb.append(tp)
            print(f"{tp : .2f}")

    # code to plot
    """
    x1 = np.asarray(Lambda)
    y1 = np.asarray(throughput_pp)
    plt.plot(x1, y1, label="pp")

    x2 = np.asarray(Lambda)
    y2 = np.asarray(throughput_op)
    plt.plot(x2, y2, label="op")

    x3 = np.asarray(Lambda)
    y3 = np.asarray(throughput_beb)
    plt.plot(x3, y3, label="beb")

    x4 = np.asarray(Lambda)
    y4 = np.asarray(throughput_lb)
    plt.plot(x4, y4, label="lb")

    plt.xlabel("Offered Load (Lambda * N)")
    plt.ylabel("Achieved Throughput (Fraction of Successful Slots)")
    plt.title("Ethernet Simulation Plot")
    plt.legend()
    plt.show()
    """


if __name__ == '__main__':
    main()
