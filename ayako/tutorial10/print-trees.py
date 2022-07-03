#!/usr/bin/python
from nltk.tree import Tree
import sys

# A program to display parse trees (in Penn treebank format) with NLTK
#
#  To install NLTK on ubuntu: sudo apt-get install python-nltk

for line in sys.stdin:
    t = Tree.fromstring(line)
    t.pretty_print()

"""
1文目の結果
(S (PP (IN Among) (NP (DT these) (NP' (, ,) (NP' (JJ supervised) (NP' (NN learning) (NNS approaches)))))) (S' (VP (VBP have) (VP (VBN been) (VP' (NP (DT the) (NP' (ADJP (RBS most) (JJ successful)) (NNS algorithms))) (PP (TO to) (NP_NN date))))) (. .)))

                                                             S                                                        
              _______________________________________________|_____________________                                    
             |                                                                     S'                                 
             |                                                _____________________|________________________________   
             |                                               VP                                                     | 
             |                                           ____|____                                                  |  
             PP                                         |         VP                                                | 
   __________|______                                    |     ____|________________                                 |  
  |                 NP                                  |    |                    VP'                               | 
  |      ___________|_______                            |    |              _______|______________________          |  
  |     |                  NP'                          |    |             NP                             |         | 
  |     |     ______________|_____                      |    |     ________|_______                       |         |  
  |     |    |                   NP'                    |    |    |               NP'                     |         | 
  |     |    |       _____________|______               |    |    |         _______|__________            |         |  
  |     |    |      |                   NP'             |    |    |       ADJP                |           PP        | 
  |     |    |      |              ______|______        |    |    |    ____|_______           |        ___|____     |  
  IN    DT   ,      JJ            NN           NNS     VBP  VBN   DT RBS           JJ        NNS      TO     NP_NN  . 
  |     |    |      |             |             |       |    |    |   |            |          |       |        |    |  
Among these  ,  supervised     learning     approaches have been the most      successful algorithms  to      date  . 
"""