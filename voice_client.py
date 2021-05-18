import requests

# Session
sess = requests.Session()

# Setting
ip_address = 'http://0.0.0.0:9999'
path_dir = './'
wav_name = 'test_dis.wav'

# Request
req = sess.post(ip_address, data={'path_dir': path_dir, 'wav_name': wav_name})

if req.status_code == 200:
    print(req,'success')
    print(req.content)
else:
    print('fail')
