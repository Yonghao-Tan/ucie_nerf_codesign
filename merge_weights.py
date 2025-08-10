import torch

net_ibr = torch.load('./pretrained/model_255000.pth')
net_ibr = torch.load('./pretrained/model_16_32.pth')
net_sr = torch.load('./pretrained/epoch896_OmniSR.pth')
net_ibr['sr_net'] = {}
for key in net_sr.keys():
    net_ibr['sr_net'][key] = net_sr[key]
print(net_ibr.keys())
torch.save(net_ibr, './pretrained/model_16_32_sr.pth')