""" 马尔科夫链预测 测试 """
from Models.forecast import markov_chain

transition_matrix = [[0.7, 0.3, 0.0],
                     [0.3, 0.1, 0.6],
                     [0.0, 0.6, 0.4]]

states = ['晴天', '阴天', '暴雨']

print('未来十天变化:')
print(markov_chain(transition_matrix, states, '晴天', 10))
