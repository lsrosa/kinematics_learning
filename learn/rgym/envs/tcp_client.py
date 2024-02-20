import socket, sys

class Client(object):
	def __init__(self,host,port):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
		
		self.msg_out = ""
		self.msg_in = ""
		self.ready2send = True
		self.messageFromGazebo = 0
		self.sent = False
		# connect to remote host
		try :
			self.s.connect((host, port))
		except :
			print('Unable to connect')
			sys.exit(0)

		print('Connected to remote host.\nType "quit" to close the connection.')
		
	def send_command(self):
		self.msg_in = ""
		sent=False
		while self.msg_in.count("_")<1:
			if not sent:
				self.s.sendall((self.msg_out+"\n").encode())
				sent=True
			data = self.s.recv(4096).decode()
			self.msg_in += data
		return