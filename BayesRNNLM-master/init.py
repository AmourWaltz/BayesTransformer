import torch
import torch.nn as nn
m = nn.Linear(3,1)
prior = m.state_dict()
prior_swp = {k:v.clone() for k,v in prior.items()}
#prior_swp = prior_swp.copy()
print("initial prior",prior)
m = nn.Linear(3,1)
m_state = m.state_dict()
m_swp = {k:v.clone() for k, v in m_state.items()}
print("initial m",m_state)
#print("before load prior", m_swp)
m_state.update(prior)
#print([ m_state[k]=v for k,v in prior.items() if k in m_state])

#print("After load prior",m_swp)
m_state['weight'][0,0]=m_swp['weight'][0,0]

print(m_state)
#prior['weight'][0,0]=0
#print(prior)
print(prior_swp)
