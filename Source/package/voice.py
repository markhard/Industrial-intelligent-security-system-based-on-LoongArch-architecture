import serial
import time
ser = serial.Serial('/dev/ttyS1', 9600)

def Voice_broadcast(data_frame):


	frame_start = bytes([0xAA, 0x55])  # 帧头
	frame_end = bytes([0x55, 0xAA])    # 帧尾

	print(bytes([data_frame]))

	data_frame = bytes([data_frame])

	complete_frame = frame_start + data_frame + frame_end

	ser.write(complete_frame)

def Voice_close():
	time.sleep(1)
	ser.close()

if __name__ == '__main__':
	Voice_broadcast(7)
	Voice_close()

